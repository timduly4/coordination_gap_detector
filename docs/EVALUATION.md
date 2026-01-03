# Evaluation Methodology Guide

## Overview

This guide covers how to evaluate search quality in the Coordination Gap Detector system. Currently, the system supports manual testing and comparison of different ranking strategies (semantic, BM25, hybrid).

**Note:** Automated evaluation pipelines, A/B testing frameworks, and continuous evaluation infrastructure are planned for future milestones but not yet implemented.

## Table of Contents

1. [Manual Search Testing](#manual-search-testing)
2. [Understanding Search Metrics](#understanding-search-metrics)
3. [Comparing Ranking Strategies](#comparing-ranking-strategies)
4. [Future Evaluation Plans](#future-evaluation-plans)
5. [Further Reading](#further-reading)

---

## Manual Search Testing

### Testing Search Quality

Test search quality by running queries and reviewing results:

```bash
# Start the system
docker compose up -d

# Load test data
docker compose exec api python scripts/generate_mock_data.py --scenarios oauth_duplication --clear

# Test semantic search
curl -X POST http://localhost:8000/api/v1/search/ \
  -H "Content-Type: application/json" \
  -d '{
    "query": "OAuth implementation decisions",
    "ranking_strategy": "semantic",
    "limit": 5
  }' | jq '.results[] | {channel, author, score, content: .content[:80]}'

# Test BM25 search
curl -X POST http://localhost:8000/api/v1/search/ \
  -H "Content-Type: application/json" \
  -d '{
    "query": "OAuth implementation decisions",
    "ranking_strategy": "bm25",
    "limit": 5
  }' | jq '.results[] | {channel, author, score, content: .content[:80]}'

# Test hybrid search
curl -X POST http://localhost:8000/api/v1/search/ \
  -H "Content-Type: application/json" \
  -d '{
    "query": "OAuth implementation decisions",
    "ranking_strategy": "hybrid_rrf",
    "limit": 5
  }' | jq '.results[] | {channel, author, score, content: .content[:80]}'
```

### Evaluation Checklist

When testing a query, ask:

- ✅ **Relevance**: Are the top results actually relevant to the query?
- ✅ **Ranking**: Are more relevant results ranked higher?
- ✅ **Coverage**: Are all important results found?
- ✅ **Diversity**: Do results cover different aspects of the topic?
- ✅ **Recency**: When appropriate, are recent results prioritized?

---

## Understanding Search Metrics

### Mean Reciprocal Rank (MRR)

**What it measures**: Position of the first relevant result

**Formula**: `MRR = (1/|Q|) · Σ 1/rank_i`

**Interpretation**:
- MRR = 1.0: First result is always relevant (perfect)
- MRR = 0.5: First relevant result is at position 2 on average
- MRR = 0.33: First relevant result is at position 3 on average

**When to use**: Single-answer queries where users want one specific result

**Example**:
```
Query 1: First relevant at rank 1 → RR = 1.0
Query 2: First relevant at rank 2 → RR = 0.5
Query 3: First relevant at rank 5 → RR = 0.2

MRR = (1.0 + 0.5 + 0.2) / 3 = 0.57
```

### Normalized Discounted Cumulative Gain (NDCG)

**What it measures**: Quality of the entire ranking, considering graded relevance

**Interpretation**:
- NDCG@10 = 1.0: Perfect ranking of top 10 results
- NDCG@10 = 0.8: Good ranking with minor issues
- NDCG@10 = 0.5: Many irrelevant or poorly ranked results

**When to use**: Multi-answer queries where users examine several results

**Key concept**: Results are graded on a scale (e.g., 0-3) and position matters - highly relevant results at the top contribute more to the score than those lower down.

### Precision@k and Recall@k

**Precision@k**: Fraction of top-k results that are relevant
- P@5 = 0.8 means 4 out of top 5 results are relevant

**Recall@k**: Fraction of all relevant documents found in top-k
- R@10 = 0.6 means top 10 results contain 60% of all relevant documents

**Trade-off**: High precision (few irrelevant results) vs. high recall (find all relevant results)

---

## Comparing Ranking Strategies

The system supports three ranking strategies:

### 1. Semantic Search
- Uses embeddings to understand query meaning
- Good for: Conceptual queries, synonyms, paraphrasing
- Limitations: May miss exact technical terms

### 2. BM25 Keyword Search
- Uses statistical keyword matching
- Good for: Exact terms, technical jargon, acronyms
- Limitations: No semantic understanding

### 3. Hybrid (RRF)
- Combines semantic + BM25 using Reciprocal Rank Fusion
- Good for: Most queries - balances both approaches
- Generally recommended as default

### Manual Comparison

Test the same query with all three strategies and compare:

```bash
# Create comparison script
cat > /tmp/compare_strategies.sh << 'EOF'
#!/bin/bash
QUERY="$1"

echo "=== SEMANTIC ==="
curl -s -X POST http://localhost:8000/api/v1/search/ \
  -H "Content-Type: application/json" \
  -d "{\"query\": \"$QUERY\", \"ranking_strategy\": \"semantic\", \"limit\": 3}" \
  | jq -r '.results[] | "[\(.score | tonumber | . * 100 | round / 100)] \(.content[:60])..."'

echo -e "\n=== BM25 ==="
curl -s -X POST http://localhost:8000/api/v1/search/ \
  -H "Content-Type: application/json" \
  -d "{\"query\": \"$QUERY\", \"ranking_strategy\": \"bm25\", \"limit\": 3}" \
  | jq -r '.results[] | "[\(.score | tonumber | . * 100 | round / 100)] \(.content[:60])..."'

echo -e "\n=== HYBRID ==="
curl -s -X POST http://localhost:8000/api/v1/search/ \
  -H "Content-Type: application/json" \
  -d "{\"query\": \"$QUERY\", \"ranking_strategy\": \"hybrid_rrf\", \"limit\": 3}" \
  | jq -r '.results[] | "[\(.score | tonumber | . * 100 | round / 100)] \(.content[:60])..."'
EOF

chmod +x /tmp/compare_strategies.sh
/tmp/compare_strategies.sh "OAuth implementation"
```

---

## Future Evaluation Plans

The following evaluation features are planned but not yet implemented:

### Automated Evaluation
- Test query sets with relevance judgments
- Automated metrics calculation (MRR, NDCG, P@k, R@k)
- Regression testing to prevent quality degradation

### A/B Testing Framework
- Online experimentation with real users
- Statistical significance testing
- Gradual rollout of ranking improvements

### Continuous Monitoring
- Real-time quality metrics dashboards
- Alerting for metric degradation
- Per-category performance tracking

---

## Further Reading

### Information Retrieval Evaluation
- **Manning et al., "Introduction to Information Retrieval", Chapter 8**
  [https://nlp.stanford.edu/IR-book/html/htmledition/evaluation-in-information-retrieval-1.html](https://nlp.stanford.edu/IR-book/html/htmledition/evaluation-in-information-retrieval-1.html)

### Ranking Metrics
- **NDCG Explained**: Järvelin & Kekäläinen (2002)
  [https://dl.acm.org/doi/10.1145/582415.582418](https://dl.acm.org/doi/10.1145/582415.582418)

### A/B Testing (Future Implementation)
- **Kohavi & Longbotham, "Online Controlled Experiments at Large Scale"**
  [https://exp-platform.com/Documents/2013-02-CACM-ExP.pdf](https://exp-platform.com/Documents/2013-02-CACM-ExP.pdf)

### Related Documentation
- [RANKING.md](./RANKING.md) - Ranking strategy details
- [DEMO.md](./DEMO.md) - Hands-on search examples
- [API_EXAMPLES.md](./API_EXAMPLES.md) - Search API usage

---

**Last Updated**: January 2025
**Status**: Manual testing only; automated evaluation planned for future milestones
