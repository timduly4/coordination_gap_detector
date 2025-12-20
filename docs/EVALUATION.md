# Evaluation Methodology Guide

## Overview

This guide covers the complete evaluation methodology for ranking quality assessment in the Coordination Gap Detector system. Proper evaluation is essential for improving search quality, comparing ranking strategies, and ensuring the system meets user needs.

## Table of Contents

1. [Evaluation Overview](#evaluation-overview)
2. [Creating Test Queries](#creating-test-queries)
3. [Relevance Judgment Guidelines](#relevance-judgment-guidelines)
4. [Running Offline Evaluations](#running-offline-evaluations)
5. [Interpreting Metrics](#interpreting-metrics)
6. [A/B Testing Methodology](#ab-testing-methodology)
7. [Continuous Evaluation](#continuous-evaluation)
8. [Best Practices](#best-practices)

---

## Evaluation Overview

### Why Evaluate?

- **Measure quality**: Quantify how well the system performs
- **Compare strategies**: Determine which ranking approach works best
- **Track improvements**: Measure impact of changes over time
- **Identify issues**: Find failure modes and edge cases
- **Optimize parameters**: Tune BM25 k1/b, fusion weights, etc.

### Evaluation Types

| Type | When | Frequency | Metrics |
|------|------|-----------|---------|
| **Offline** | Before deployment | Continuous | MRR, NDCG, P@k, R@k |
| **Online** | In production | Real-time | CTR, Time-to-success, Abandonment |
| **User Studies** | Major changes | Quarterly | Task success, Satisfaction |

### Evaluation Workflow

```
1. Create Test Queries ──► 2. Collect Judgments ──► 3. Run Evaluation ──► 4. Analyze Results
       │                           │                         │                    │
       ↓                           ↓                         ↓                    ↓
   Query types           Relevance scale               Metrics            Statistical tests
   Coverage              Inter-rater                   Baselines           A/B decisions
   Diversity             agreement                     Comparison          Improvements
```

---

## Creating Test Queries

### Query Set Requirements

A good test query set should have:

- **Size**: Minimum 50 queries, ideally 200+
- **Coverage**: Representative of real user queries
- **Diversity**: Different query types, lengths, topics
- **Difficulty**: Mix of easy and hard queries
- **Balance**: Cover all major use cases

### Query Categories

#### 1. Factual Queries

Seeking specific information or facts.

**Examples**:
```
- "Who approved the OAuth decision?"
- "What database did we choose?"
- "When was the API migration completed?"
```

**Characteristics**:
- Single correct answer
- High precision required
- First result matters most (MRR important)

#### 2. Technical Queries

Looking for code, implementations, or technical details.

**Examples**:
```
- "OAuth token validation code"
- "API rate limiting implementation"
- "Database migration script errors"
```

**Characteristics**:
- Exact keyword matches important
- BM25 often performs well
- Code snippets valuable

#### 3. Temporal Queries

Time-sensitive or recency-based queries.

**Examples**:
```
- "Recent discussions about authentication"
- "Latest OAuth security updates"
- "Current API design decisions"
```

**Characteristics**:
- Recency matters
- Need freshness signals
- Older results may be obsolete

#### 4. Multi-Source Queries

Require information from multiple channels or sources.

**Examples**:
```
- "OAuth mentioned in Slack and GitHub"
- "Authentication discussed across teams"
- "Security decisions in all channels"
```

**Characteristics**:
- Cross-source evidence needed
- Multiple relevant results
- NDCG more important than MRR

#### 5. Ambiguous Queries

Queries with multiple interpretations.

**Examples**:
```
- "Login issues" (user login? admin? OAuth? session?)
- "API problems" (rate limits? errors? design?)
- "Migration" (database? cloud? authentication?)
```

**Characteristics**:
- Challenging for all methods
- Need disambiguation
- Diversity in results helpful

### Generating Test Queries

#### Method 1: Sample from Logs

```bash
# Extract real queries from logs (if available)
python scripts/generate_test_queries.py \
  --source logs \
  --sample-rate 0.1 \
  --min-frequency 3 \
  --output data/test_queries/sampled_queries.jsonl
```

#### Method 2: Synthetic Generation

```bash
# Generate queries from mock data
python scripts/generate_test_queries.py \
  --scenarios all \
  --queries-per-scenario 10 \
  --output data/test_queries/generated_queries.jsonl
```

#### Method 3: Manual Curation

Create manually to ensure coverage:

```json
{
  "query_id": "oauth_impl_001",
  "query": "OAuth implementation decisions",
  "category": "factual",
  "difficulty": "medium",
  "expected_source_types": ["slack", "google_docs"],
  "notes": "Should find architecture discussions"
}
```

### Query Set Validation

Check your test set quality:

```python
from src.ranking.evaluation import analyze_query_set

analysis = analyze_query_set("data/test_queries/queries.jsonl")

print(f"Total queries: {analysis['total']}")
print(f"Average length: {analysis['avg_length']} words")
print(f"Categories: {analysis['categories']}")
print(f"Difficulty distribution: {analysis['difficulty']}")
```

**Quality Checklist**:
- [ ] At least 50 queries
- [ ] All categories represented
- [ ] Mix of query lengths (1-10+ words)
- [ ] Mix of difficulty levels
- [ ] Real or realistic queries
- [ ] Cover main use cases

---

## Relevance Judgment Guidelines

### Relevance Scale

Use a 4-point scale (0-3):

| Score | Label | Meaning | When to Use |
|-------|-------|---------|-------------|
| **3** | Highly Relevant | Perfect match, directly answers query | - Exact answer to question<br>- Discusses precise topic<br>- Authoritative source |
| **2** | Relevant | Good match, helpful but not perfect | - Partially answers query<br>- Related discussion<br>- Supporting information |
| **1** | Partially Relevant | Tangentially related, low value | - Mentions topic briefly<br>- Related but different focus<br>- Background information |
| **0** | Not Relevant | Unrelated, no value | - Different topic<br>- Misleading<br>- Duplicate content |

### Judgment Examples

#### Query: "OAuth implementation decisions"

**Score 3 (Highly Relevant)**:
```
"OAuth Implementation Decision

After evaluating Auth0, Okta, and self-hosted options, we've decided
to use Auth0 for OAuth2 implementation. Key reasons:
1. Enterprise support and SLA
2. SAML compatibility needed for SSO
3. Extensive documentation and libraries
4. Cost effective at our scale

Decision approved by: Architecture team, Security team
Implementation timeline: Q1 2024"
```

**Score 2 (Relevant)**:
```
"Starting OAuth implementation in auth-service this week. Following
the architecture decision from last month. Will use the Auth0 patterns
discussed in #architecture."
```

**Score 1 (Partially Relevant)**:
```
"Fixed a bug in the login flow. Wasn't OAuth related, just a session
cookie configuration issue. But good to test OAuth integration too."
```

**Score 0 (Not Relevant)**:
```
"Team meeting scheduled for tomorrow at 2pm. Agenda includes project
updates and sprint planning."
```

### Collecting Judgments

#### Format

Store judgments in JSONL format:

```json
{
  "query_id": "oauth_impl_001",
  "query_text": "OAuth implementation decisions",
  "document_id": "msg_slack_12345",
  "relevance": 3,
  "judged_by": "evaluator_1",
  "judged_at": "2024-01-15T10:30:00Z",
  "notes": "Perfect match - architecture decision document"
}
```

#### Tools

Use the evaluation service API:

```python
from src.services.evaluation_service import EvaluationService

eval_service = EvaluationService(search_service)

# Add judgment
eval_service.add_judgment(
    query_id="oauth_impl_001",
    document_id="msg_slack_12345",
    relevance=3,
    query_text="OAuth implementation decisions"
)
```

Or use the interactive judging tool:

```bash
python scripts/interactive_judging.py \
  --queries data/test_queries/queries.jsonl \
  --output data/test_queries/judgments.jsonl
```

### Inter-Rater Reliability

With multiple judges, measure agreement:

```python
from src.ranking.evaluation import calculate_inter_rater_agreement

agreement = calculate_inter_rater_agreement(
    judgments_file="data/test_queries/judgments.jsonl",
    metric="cohen_kappa"  # or "fleiss_kappa" for 3+ raters
)

print(f"Inter-rater agreement (Cohen's κ): {agreement:.3f}")

# κ > 0.8: Excellent agreement
# κ > 0.6: Good agreement
# κ > 0.4: Moderate agreement
# κ < 0.4: Poor agreement (need clearer guidelines)
```

### Quality Control

Review judgments for consistency:

```python
from src.ranking.evaluation import check_judgment_quality

issues = check_judgment_quality("data/test_queries/judgments.jsonl")

for issue in issues:
    print(f"Query {issue['query_id']}: {issue['problem']}")
    # Example: "Same document rated 3 and 0 by different judges"
```

---

## Running Offline Evaluations

### Basic Evaluation

Evaluate a single strategy:

```bash
python scripts/evaluate_ranking.py \
  --queries data/test_queries/queries.jsonl \
  --judgments data/test_queries/judgments.jsonl \
  --strategy semantic \
  --metrics mrr,ndcg@10,p@5,r@10 \
  --output results/semantic_eval.json
```

Output:
```
Evaluation Results (semantic)
═════════════════════════════════
MRR:        0.742
NDCG@10:    0.789
P@5:        0.680
R@10:       0.583

Per-query breakdown:
┌────────────┬───────┬──────────┬──────┐
│ Query      │  MRR  │ NDCG@10  │  P@5 │
├────────────┼───────┼──────────┼──────┤
│ oauth_001  │ 1.000 │  0.923   │ 0.80 │
│ oauth_002  │ 0.500 │  0.756   │ 0.60 │
│ auth_001   │ 0.333 │  0.612   │ 0.40 │
...
```

### Comparing Strategies

Compare multiple ranking strategies:

```bash
python scripts/evaluate_ranking.py \
  --queries data/test_queries/queries.jsonl \
  --judgments data/test_queries/judgments.jsonl \
  --strategies semantic,bm25,hybrid_rrf,hybrid_weighted \
  --metrics mrr,ndcg@10,p@5 \
  --output results/strategy_comparison.json
```

Output:
```
Strategy Comparison
═════════════════════════════════════════════════════════
┌─────────────────┬───────┬──────────┬──────────┬──────────┐
│ Strategy        │  MRR  │ NDCG@10  │  P@5     │  R@10    │
├─────────────────┼───────┼──────────┼──────────┼──────────┤
│ semantic        │ 0.742 │  0.789   │  0.680   │  0.583   │
│ bm25            │ 0.685 │  0.721   │  0.620   │  0.542   │
│ hybrid_rrf      │ 0.798 │  0.841   │  0.740   │  0.645   │ ⭐ Best
│ hybrid_weighted │ 0.776 │  0.823   │  0.720   │  0.628   │
└─────────────────┴───────┴──────────┴──────────┴──────────┘

Improvements over semantic:
- hybrid_rrf: +7.5% MRR, +6.6% NDCG@10
- Statistical significance: p < 0.01 (t-test)

Best strategy: hybrid_rrf
```

### Advanced Evaluation

With statistical significance testing:

```bash
python scripts/evaluate_ranking.py \
  --queries data/test_queries/queries.jsonl \
  --judgments data/test_queries/judgments.jsonl \
  --strategies semantic,hybrid_rrf \
  --metrics mrr,ndcg@10 \
  --statistical-test t-test \
  --significance-level 0.05 \
  --output results/advanced_eval.json
```

Output includes:
```
Statistical Significance (hybrid_rrf vs semantic)
═══════════════════════════════════════════════
Metric:  MRR
Diff:    +0.056 (+7.5%)
p-value: 0.003
Result:  ✓ Statistically significant at α=0.05

Metric:  NDCG@10
Diff:    +0.052 (+6.6%)
p-value: 0.008
Result:  ✓ Statistically significant at α=0.05

Conclusion: hybrid_rrf is significantly better than semantic
```

### Per-Category Analysis

Break down by query category:

```bash
python scripts/evaluate_ranking.py \
  --queries data/test_queries/queries.jsonl \
  --judgments data/test_queries/judgments.jsonl \
  --strategies semantic,bm25,hybrid_rrf \
  --metrics mrr,ndcg@10 \
  --group-by category \
  --output results/category_breakdown.json
```

Output:
```
Results by Category
═══════════════════════════════════════════════════

Factual Queries (n=25)
┌─────────────┬───────┬──────────┐
│ Strategy    │  MRR  │ NDCG@10  │
├─────────────┼───────┼──────────┤
│ semantic    │ 0.820 │  0.856   │
│ bm25        │ 0.740 │  0.782   │
│ hybrid_rrf  │ 0.880 │  0.911   │ ⭐
└─────────────┴───────┴──────────┘

Technical Queries (n=30)
┌─────────────┬───────┬──────────┐
│ Strategy    │  MRR  │ NDCG@10  │
├─────────────┼───────┼──────────┤
│ semantic    │ 0.683 │  0.721   │
│ bm25        │ 0.745 │  0.789   │ ⭐
│ hybrid_rrf  │ 0.792 │  0.831   │
└─────────────┴───────┴──────────┘

Insights:
- BM25 strong for technical queries (exact matches)
- Semantic strong for factual queries (understanding)
- Hybrid best overall across categories
```

---

## Interpreting Metrics

### Mean Reciprocal Rank (MRR)

**Formula**: `MRR = (1/|Q|) · Σ 1/rank_i`

**Interpretation**:
- Measures rank of **first** relevant result
- Range: [0, 1], higher is better
- 1.0 = first result always relevant

**Use When**:
- Users typically click only one result
- Navigational queries (finding specific thing)
- First impression matters

**Example**:
```
Query 1: First relevant at rank 1 → RR = 1.0
Query 2: First relevant at rank 2 → RR = 0.5
Query 3: First relevant at rank 5 → RR = 0.2
Query 4: No relevant results      → RR = 0.0

MRR = (1.0 + 0.5 + 0.2 + 0.0) / 4 = 0.425
```

**What's Good**:
- MRR > 0.8: Excellent (top result usually relevant)
- MRR > 0.6: Good
- MRR > 0.4: Fair
- MRR < 0.4: Poor (needs improvement)

### Normalized Discounted Cumulative Gain (NDCG)

**Formula**:
```
DCG@k = Σ (2^rel_i - 1) / log2(i + 1)
NDCG@k = DCG@k / iDCG@k
```

**Interpretation**:
- Measures quality of **entire ranking**
- Accounts for graded relevance (0-3 scale)
- Position-aware (top results matter more)
- Range: [0, 1], higher is better

**Use When**:
- Users examine multiple results
- Relevance is graded (not binary)
- Ranking quality throughout list matters

**Example**:
```
Results: [3, 2, 2, 1, 0, 0]  (relevance scores)

DCG@6 = (2^3-1)/log2(2) + (2^2-1)/log2(3) + (2^2-1)/log2(4) +
        (2^1-1)/log2(5) + 0 + 0
      = 7.0 + 1.89 + 1.5 + 0.43
      = 10.82

Ideal: [3, 3, 2, 2, 1, 0]
iDCG@6 = 11.71

NDCG@6 = 10.82 / 11.71 = 0.924
```

**What's Good**:
- NDCG@10 > 0.8: Excellent
- NDCG@10 > 0.6: Good
- NDCG@10 > 0.4: Fair
- NDCG@10 < 0.4: Poor

### Precision@k and Recall@k

**Precision@k**: Fraction of top-k results that are relevant
```
P@k = (# relevant in top k) / k
```

**Recall@k**: Fraction of all relevant docs found in top-k
```
R@k = (# relevant in top k) / (total relevant)
```

**Use When**:
- Binary relevance (relevant or not)
- Want to measure coverage
- Trade-off between precision and recall

**Example**:
```
Top 5 results: [R, R, N, R, N]  (R=relevant, N=not relevant)
Total relevant in collection: 10

P@5 = 3/5 = 0.60
R@5 = 3/10 = 0.30
```

### Which Metric to Use?

| Scenario | Primary Metric | Secondary Metrics |
|----------|----------------|-------------------|
| Single answer queries | MRR | P@1, Success@5 |
| Multiple good answers | NDCG@10 | MRR, P@5 |
| Comprehensive results | R@20, NDCG@20 | P@10 |
| Comparing strategies | NDCG@10, MRR | P@5, R@10 |
| Production monitoring | MRR, CTR | Time-to-click |

### Metric Baselines

Establish baselines for your domain:

```
Strong Performance:
- MRR > 0.75
- NDCG@10 > 0.80
- P@5 > 0.70
- R@10 > 0.60

Acceptable Performance:
- MRR > 0.60
- NDCG@10 > 0.65
- P@5 > 0.55
- R@10 > 0.45

Needs Improvement:
- MRR < 0.50
- NDCG@10 < 0.55
- P@5 < 0.45
- R@10 < 0.35
```

---

## A/B Testing Methodology

### When to A/B Test

- Deploying new ranking strategy
- Changing parameters (BM25 k1/b, fusion weights)
- Adding new features
- Significant algorithm changes

### Setting Up A/B Tests

```python
from src.ranking.experiments import ABTest

# Create A/B test
test = ABTest(
    name="hybrid_weight_optimization",
    control="hybrid_rrf",  # Current production
    treatment="hybrid_weighted_80_20",  # New variant
    traffic_split=0.5,  # 50/50 split
    duration_days=14
)

# Assign users to variants
variant = test.assign_user(user_id="user_123")
```

### Sample Size Calculation

Determine required sample size:

```python
from src.ranking.experiments import calculate_sample_size

sample_size = calculate_sample_size(
    baseline_metric=0.75,  # Current MRR
    minimum_detectable_effect=0.05,  # Want to detect 5% improvement
    statistical_power=0.80,
    significance_level=0.05
)

print(f"Need {sample_size} queries per variant")
# Example output: "Need 1250 queries per variant"
```

### Running the Test

Monitor metrics during test:

```python
# Daily monitoring
results = test.get_current_results()

print(f"Control MRR: {results['control']['mrr']:.3f}")
print(f"Treatment MRR: {results['treatment']['mrr']:.3f}")
print(f"Lift: {results['lift_pct']:.1f}%")
print(f"Confidence: {results['confidence']:.1f}%")
```

### Analyzing Results

At end of test period:

```python
final_results = test.analyze(
    significance_level=0.05,
    correction="bonferroni"  # For multiple metrics
)

print(final_results.summary())
```

Output:
```
A/B Test Results: hybrid_weight_optimization
═══════════════════════════════════════════════

Duration: 14 days
Queries: 3,247 (control), 3,198 (treatment)

Metrics:
┌────────────┬─────────┬───────────┬────────┬───────────┬────────────┐
│ Metric     │ Control │ Treatment │ Lift   │ p-value   │ Significant│
├────────────┼─────────┼───────────┼────────┼───────────┼────────────┤
│ MRR        │  0.742  │   0.769   │ +3.6%  │  0.023    │ ✓ Yes      │
│ NDCG@10    │  0.789  │   0.804   │ +1.9%  │  0.128    │ ✗ No       │
│ CTR        │  0.651  │   0.673   │ +3.4%  │  0.041    │ ✓ Yes      │
└────────────┴─────────┴───────────┴────────┴───────────┴────────────┘

Decision: SHIP IT ✓
- MRR improved significantly (+3.6%)
- CTR improved significantly (+3.4%)
- No negative metrics
- Recommend deploying treatment to 100%
```

### Common Pitfalls

**1. Peeking Too Early**
```python
# ❌ Wrong: Check every day and stop when significant
if test.is_significant():
    test.stop()  # Inflates false positive rate!

# ✓ Correct: Wait for planned duration
if test.days_running >= test.planned_duration:
    results = test.analyze()
```

**2. Multiple Testing**
```python
# ❌ Wrong: Test 10 metrics, use first that's significant
# Inflates false positive rate

# ✓ Correct: Bonferroni correction
results = test.analyze(
    metrics=["mrr", "ndcg", "ctr"],
    correction="bonferroni"
)
```

**3. Insufficient Sample Size**
```python
# ❌ Wrong: Run test with 100 queries
# Not enough power to detect realistic improvements

# ✓ Correct: Calculate required sample size first
min_sample = calculate_sample_size(...)
print(f"Need at least {min_sample} queries")
```

---

## Continuous Evaluation

### Automated Evaluation Pipeline

Set up continuous evaluation:

```bash
# Run nightly evaluation
0 2 * * * /usr/bin/python scripts/evaluate_ranking.py \
  --queries data/test_queries/queries.jsonl \
  --judgments data/test_queries/judgments.jsonl \
  --strategies production \
  --metrics mrr,ndcg@10 \
  --output results/nightly/$(date +\%Y\%m\%d).json \
  --alert-if-below mrr:0.70,ndcg:0.75
```

### Quality Monitoring Dashboard

Track metrics over time:

```python
from src.infrastructure.observability import track_metric

# After each query
track_metric(
    name="search.mrr",
    value=calculated_mrr,
    labels={"strategy": "hybrid_rrf", "category": query_category}
)
```

Visualize in Grafana:
- MRR over time
- NDCG@10 trends
- Per-category breakdowns
- Strategy comparisons

### Regression Testing

Prevent quality degradation:

```python
# In CI/CD pipeline
from src.ranking.evaluation import run_regression_tests

results = run_regression_tests(
    test_set="data/test_queries/regression_suite.jsonl",
    current_strategy="production"
)

if results["mrr"] < 0.70:
    raise Exception("MRR regression detected!")
    # Block deployment
```

---

## Best Practices

### 1. Start Small, Iterate

```
Phase 1: 50 queries, manual judgments
Phase 2: 100 queries, measure improvements
Phase 3: 200+ queries, comprehensive coverage
```

### 2. Diverse Test Set

Ensure queries cover:
- All categories (factual, technical, temporal, etc.)
- Different lengths (1-15+ words)
- Easy, medium, hard queries
- Common and rare queries
- Successful and failed queries

### 3. Fresh Judgments

```python
# Re-judge periodically (every 3-6 months)
# - System improves: old judgments may be stale
# - Content changes: relevance shifts over time
# - Understanding evolves: better judgment criteria
```

### 4. Multiple Metrics

Don't optimize for single metric:

```python
# ❌ Only MRR → May sacrifice overall ranking quality
# ✓ MRR + NDCG@10 + P@5 → Balanced optimization
```

### 5. Segment Analysis

Break down by query characteristics:

```python
# By category
eval.analyze_by_category()

# By query length
eval.analyze_by_length_bucket()

# By query frequency
eval.analyze_by_popularity()
```

### 6. Error Analysis

Investigate failures:

```python
from src.ranking.evaluation import find_failure_cases

failures = find_failure_cases(
    results_file="results/eval.json",
    threshold_mrr=0.3  # Queries with MRR < 0.3
)

for query, result in failures:
    print(f"Query: {query}")
    print(f"MRR: {result['mrr']:.3f}")
    print(f"Top result: {result['top_doc']}")
    print(f"Issue: {diagnose_failure(query, result)}")
    print()
```

### 7. Document Everything

Maintain evaluation documentation:

```markdown
# Evaluation Log

## 2024-01-15: Baseline Evaluation
- Test set: 50 queries
- Strategy: semantic
- Results: MRR=0.742, NDCG@10=0.789

## 2024-01-20: Hybrid RRF vs Semantic
- Added hybrid_rrf strategy
- Results: MRR improved +7.5% (p<0.01)
- Decision: Deploy hybrid_rrf

## 2024-02-01: Weight Tuning
- Tested weights [0.5/0.5, 0.7/0.3, 0.8/0.2]
- Best: 0.7/0.3 (MRR=0.798)
- Decision: Keep current weights
```

---

## Troubleshooting Evaluation

### Low Baseline Metrics

**Symptoms**: MRR < 0.50, NDCG@10 < 0.55

**Possible Causes**:
- Test queries too hard
- Relevance judgments too strict
- System genuinely underperforming
- Insufficient data indexed

**Solutions**:
```python
# Check query difficulty
analyze_query_difficulty(queries)

# Review judgment criteria
check_judgment_distribution()  # Should have some 2s and 3s

# Verify data
check_index_coverage()  # Is relevant content indexed?
```

### High Variance Across Queries

**Symptoms**: Some queries perfect (MRR=1.0), others terrible (MRR=0.0)

**Solutions**:
```python
# Identify outliers
outliers = find_outlier_queries(results)

# Analyze patterns
for query in outliers:
    print(f"{query}: {diagnose_issue(query)}")

# May need query-specific handling
```

### Metrics Don't Match User Experience

**Symptoms**: Offline metrics good, but users complaining

**Possible Causes**:
- Test set not representative
- Missing important query types
- Offline/online evaluation mismatch

**Solutions**:
```python
# Sample real user queries
real_queries = sample_from_logs(n=100)

# Compare distributions
compare_query_distributions(
    test_set=test_queries,
    real_queries=real_queries
)

# Update test set to match real usage
```

---

## Further Reading

- **IR Evaluation**: [Manning et al., Introduction to Information Retrieval, Ch. 8](https://nlp.stanford.edu/IR-book/html/htmledition/evaluation-in-information-retrieval-1.html)
- **NDCG**: [Järvelin & Kekäläinen, 2002](https://dl.acm.org/doi/10.1145/582415.582418)
- **A/B Testing**: [Kohavi & Longbotham, 2017](https://exp-platform.com/Documents/2013-02-CACM-ExP.pdf)
- **Statistical Significance**: [Smucker et al., 2007](https://dl.acm.org/doi/10.1145/1277741.1277760)

---

**Last Updated**: December 2024
**Version**: 1.0
**Related**: See [RANKING.md](./RANKING.md) for ranking strategy details
