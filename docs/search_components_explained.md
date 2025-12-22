# Search Components Explained

Understanding the four key components of the search system: **Search Infrastructure**, **Ranking Algorithms**, **Evaluation Metrics**, and **Feature Extraction**.

## Overview: The Search Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                    USER QUERY: "OAuth security"                 │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│  1. SEARCH INFRASTRUCTURE (Get Candidates)                      │
│  ┌──────────────────────┐  ┌──────────────────────┐            │
│  │ Semantic Search      │  │ Keyword Search       │            │
│  │ (ChromaDB)           │  │ (Elasticsearch)      │            │
│  │ Returns: [5,8,12]    │  │ Returns: [5,12,15]   │            │
│  └──────────────────────┘  └──────────────────────┘            │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│  2. FEATURE EXTRACTION (Compute Signals)                        │
│  For each document, extract 24 features:                        │
│  • semantic_score: 0.92                                         │
│  • bm25_score: 0.85                                             │
│  • exact_match: 1.0                                             │
│  • recency: 0.95                                                │
│  • thread_depth: 0.60                                           │
│  • author_authority: 0.75                                       │
│  ... (18 more features)                                         │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│  3. RANKING ALGORITHM (Order Results)                           │
│  Take features → Compute final score → Rank                     │
│  • RRF: 1/(60+rank_sem) + 1/(60+rank_kw)                        │
│  • Weighted: 0.7*semantic + 0.3*keyword                         │
│  • ML Model: Learned from labeled data                          │
│  Final Ranking: [5, 8, 12, 15]                                  │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│  4. EVALUATION METRICS (Measure Quality)                        │
│  How good is this ranking?                                      │
│  • MRR: 1.0 (first result is relevant)                          │
│  • NDCG@5: 0.95 (excellent ranking)                             │
│  • Precision@3: 1.0 (all top 3 relevant)                        │
└─────────────────────────────────────────────────────────────────┘
```

---

## 1. Search Infrastructure

**What it does**: Retrieves candidate documents that might be relevant to the query.

**Think of it as**: The first filter - casting a wide net to find potentially relevant items.

### Two Types of Search

#### A. Semantic Search (ChromaDB)
**Finds**: Documents with similar **meaning** to the query

```python
# Query: "user login problems"
# Returns documents about:
# - "authentication failing after password reset" ✅
# - "users can't sign in on mobile" ✅
# - "login screen stuck on loading" ✅

# How it works:
query_embedding = [0.234, -0.567, 0.891, ...]  # 768 numbers
# Compare with all document embeddings using cosine similarity
# Return top K most similar
```

**Strengths**:
- Handles synonyms ("login" = "sign-in" = "authentication")
- Understands concepts, not just keywords
- Works across languages/paraphrasing

**Weaknesses**:
- Can miss exact keyword matches
- Slower than keyword search
- Requires embedding generation

#### B. Keyword Search (Elasticsearch + BM25)
**Finds**: Documents containing specific **words** from the query

```python
# Query: "OAuth security"
# Returns documents containing:
# - "OAuth" AND "security" (highest score)
# - "OAuth" OR "security" (lower score)
# - Variations: "OAuth2", "secure" (if stemming enabled)

# How BM25 scores a document:
# 1. Term Frequency (TF): How many times does "OAuth" appear?
# 2. Inverse Document Frequency (IDF): How rare is "OAuth"?
# 3. Length Normalization: Penalize very long documents
# Score = IDF * (TF adjusted for length)
```

**Strengths**:
- Very fast (pre-built inverted index)
- Exact keyword matching
- Good for technical terms, names, IDs

**Weaknesses**:
- Doesn't understand synonyms
- Can't handle paraphrasing
- Sensitive to spelling, word forms

### Why Use Both?

**Hybrid Search** combines the strengths of both:
- Semantic finds conceptually related items
- Keyword ensures exact matches aren't missed
- Together: comprehensive, relevant results

---

## 2. Ranking Algorithms

**What it does**: Orders the candidate documents from most to least relevant.

**Think of it as**: The final decision - "Given these candidates, which order should we show them?"

### Types of Ranking Algorithms

#### A. BM25 (Keyword Ranking)
```python
# Algorithm: Probabilistic ranking based on term statistics
# Input: Query terms, document text
# Output: Relevance score

# Example from src/ranking/scoring.py:
class BM25Scorer:
    def score(self, query_terms, document, doc_length, avg_doc_length):
        score = 0
        for term in query_terms:
            # 1. Calculate IDF (how rare is this term?)
            idf = log((N - df + 0.5) / (df + 0.5))

            # 2. Calculate TF with saturation (k1) and length norm (b)
            tf = term_frequency(term, document)
            norm = (1 - b + b * (doc_length / avg_doc_length))
            tf_component = (tf * (k1 + 1)) / (tf + k1 * norm)

            # 3. Combine: score += IDF * TF_component
            score += idf * tf_component

        return score
```

**Parameters**:
- `k1=1.5`: Term frequency saturation (how much to reward repeated terms)
- `b=0.75`: Length normalization (how much to penalize long documents)

**When to use**: Keyword/exact match queries ("find message with ID oauth_123")

#### B. Reciprocal Rank Fusion (RRF)
```python
# Algorithm: Combine multiple ranked lists without score normalization
# Input: Multiple ranked lists (semantic results, keyword results)
# Output: Fused ranking

# Example from src/search/hybrid_search.py:
def reciprocal_rank_fusion(semantic_results, keyword_results, k=60):
    scores = {}

    # For each result, sum reciprocal ranks from all lists
    for rank, doc in enumerate(semantic_results, start=1):
        scores[doc.id] = scores.get(doc.id, 0) + 1/(k + rank)

    for rank, doc in enumerate(keyword_results, start=1):
        scores[doc.id] = scores.get(doc.id, 0) + 1/(k + rank)

    # Sort by combined score
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)

# Example:
# Semantic: [doc5 (rank 1), doc8 (rank 2), doc12 (rank 3)]
# Keyword:  [doc5 (rank 2), doc12 (rank 1), doc15 (rank 3)]
#
# doc5:  1/(60+1) + 1/(60+2) = 0.0164 + 0.0161 = 0.0325 ✅ (appears in both!)
# doc8:  1/(60+2) = 0.0161
# doc12: 1/(60+3) + 1/(60+1) = 0.0159 + 0.0164 = 0.0323
# doc15: 1/(60+3) = 0.0159
#
# Final ranking: [doc5, doc12, doc8, doc15]
```

**When to use**: Combining multiple search strategies without worrying about score normalization

#### C. Weighted Score Fusion
```python
# Algorithm: Linear combination of normalized scores
# Input: Multiple scoring systems with weights
# Output: Weighted average score

def weighted_fusion(semantic_score, keyword_score, w_sem=0.7, w_kw=0.3):
    # Normalize scores to [0,1] first (if needed)
    final_score = w_sem * semantic_score + w_kw * keyword_score
    return final_score

# Example:
# doc5: 0.7 * 0.92 + 0.3 * 0.85 = 0.644 + 0.255 = 0.899
# doc8: 0.7 * 0.88 + 0.3 * 0.20 = 0.616 + 0.060 = 0.676
```

**When to use**: When you trust score calibration and want explicit control over strategy weights

#### D. ML-Based Ranking (Learning to Rank)
```python
# Algorithm: Machine learning model trained on labeled data
# Input: Feature vector (24 features per document)
# Output: Predicted relevance score

# Conceptual example:
model = LambdaMART()  # Gradient boosted decision trees
model.train(
    features=[semantic_score, bm25_score, recency, thread_depth, ...],
    labels=[3, 2, 1, 0, 0, ...]  # Graded relevance judgments
)

# At inference:
relevance_score = model.predict([0.92, 0.85, 0.95, 0.60, ...])
```

**When to use**: When you have labeled training data and want to learn optimal feature weights

---

## 3. Evaluation Metrics

**What it does**: Measures how **good** your ranking is compared to ground truth.

**Think of it as**: The report card - "How well did we do?"

### Key Metrics

#### A. Mean Reciprocal Rank (MRR)

**Measures**: Position of the **first relevant result**

```python
# Example from src/ranking/metrics.py:
def calculate_mrr(queries):
    """
    MRR = Average of (1 / rank_of_first_relevant)

    Example queries:
    Query 1: [0, 1, 0, 0]  → First relevant at rank 2 → 1/2 = 0.5
    Query 2: [1, 0, 0, 0]  → First relevant at rank 1 → 1/1 = 1.0
    Query 3: [0, 0, 0, 1]  → First relevant at rank 4 → 1/4 = 0.25

    MRR = (0.5 + 1.0 + 0.25) / 3 = 0.583
    """
    reciprocal_ranks = []
    for query_results in queries:
        for i, relevance in enumerate(query_results, start=1):
            if relevance > 0:
                reciprocal_ranks.append(1.0 / i)
                break

    return sum(reciprocal_ranks) / len(reciprocal_ranks)
```

**Range**: 0 to 1 (higher is better)
- **1.0** = Perfect! First result always relevant
- **0.5** = First relevant result usually at rank 2
- **0.0** = No relevant results found

**When to use**: Navigational queries where users need ONE right answer (e.g., "find the OAuth implementation doc")

#### B. Normalized Discounted Cumulative Gain (NDCG@k)

**Measures**: Quality of the **entire ranking** with graded relevance

```python
# Example from src/ranking/metrics.py:
def calculate_ndcg(relevance_scores, k=5):
    """
    NDCG = DCG / iDCG (normalized to [0,1])

    DCG@k = Σ (2^rel_i - 1) / log2(i + 1)

    Example ranking (relevance: 0=not relevant, 3=perfect):
    Ranking:      [3, 2, 0, 1, 2]
    Ideal ranking:[3, 2, 2, 1, 0]  ← sorted descending

    DCG:
    i=1: (2^3 - 1) / log2(2) = 7.0 / 1.0   = 7.0
    i=2: (2^2 - 1) / log2(3) = 3.0 / 1.585 = 1.89
    i=3: (2^0 - 1) / log2(4) = 0.0 / 2.0   = 0.0
    i=4: (2^1 - 1) / log2(5) = 1.0 / 2.322 = 0.43
    i=5: (2^2 - 1) / log2(6) = 3.0 / 2.585 = 1.16
    DCG = 10.48

    iDCG (ideal): [3, 2, 2, 1, 0] → 10.79

    NDCG = 10.48 / 10.79 = 0.971 (excellent!)
    """
```

**Range**: 0 to 1 (higher is better)
- **1.0** = Perfect ranking (same as ideal)
- **0.9+** = Excellent
- **0.7-0.9** = Good
- **<0.5** = Poor

**When to use**: When you care about the quality of the entire result list, not just the first hit

#### C. Precision@k and Recall@k

```python
# Precision@k: What fraction of top k results are relevant?
def precision_at_k(relevance, k=3):
    """
    Example: [1, 1, 0, 1, 0] at k=3
    Top 3: [1, 1, 0]
    Precision@3 = 2/3 = 0.667
    """
    return sum(relevance[:k]) / k

# Recall@k: What fraction of all relevant items are in top k?
def recall_at_k(relevance, k=3):
    """
    Example: [1, 1, 0, 1, 0] at k=3
    Total relevant: 3
    Found in top 3: 2
    Recall@3 = 2/3 = 0.667
    """
    total_relevant = sum(relevance)
    found_relevant = sum(relevance[:k])
    return found_relevant / total_relevant if total_relevant > 0 else 0.0
```

**When to use**:
- **Precision**: When false positives are costly (showing irrelevant results)
- **Recall**: When false negatives are costly (missing relevant results)

---

## 4. Feature Extraction

**What it does**: Extracts **signals** about query-document relevance.

**Think of it as**: The evidence collector - "What do we know about this document that helps determine relevance?"

### Feature Categories (24 features total)

```python
# From src/ranking/features.py:
class FeatureExtractor:
    """Extracts 24 features across 5 categories"""
```

#### A. Query-Document Similarity (6 features)
```python
features = {
    # 1. Semantic similarity from embeddings
    "semantic_score": 0.92,  # Cosine similarity

    # 2. BM25 keyword score
    "bm25_score": 0.85,

    # 3. Exact match bonus
    "exact_match": 1.0,  # Does document contain exact query phrase?

    # 4. Query term coverage
    "term_coverage": 0.75,  # What fraction of query terms appear?

    # 5. Longest common subsequence
    "lcs_ratio": 0.60,

    # 6. Edit distance
    "edit_distance_normalized": 0.95
}
```

#### B. Temporal Signals (5 features)
```python
features = {
    # 1. How recent is this document?
    "recency": 0.95,  # 1.0 = today, decays exponentially

    # 2. Is this in an active discussion?
    "activity_burst": 1.0,  # Spike in recent messages?

    # 3. How fresh was the last edit?
    "edit_freshness": 0.80,

    # 4. Response velocity (how fast are people replying?)
    "response_velocity": 0.70,

    # 5. Temporal alignment with query
    "temporal_relevance": 0.85  # If query mentions "yesterday"
}
```

#### C. Engagement Metrics (6 features)
```python
features = {
    # 1. How deep is the thread?
    "thread_depth": 0.60,  # 0-100 messages → normalized

    # 2. Number of participants
    "participant_count": 0.45,

    # 3. Reaction count
    "reaction_count": 0.30,

    # 4. Reply count
    "reply_count": 0.55,

    # 5. View count (if available)
    "view_count": 0.40,

    # 6. Share/forward count
    "share_count": 0.20
}
```

#### D. Source Authority (4 features)
```python
features = {
    # 1. Author's organizational rank/influence
    "author_authority": 0.75,

    # 2. Channel importance
    "channel_authority": 0.85,  # #announcements > #random

    # 3. Domain expertise
    "domain_expertise": 0.90,  # Is author an expert in this area?

    # 4. Past contribution quality
    "contribution_score": 0.80
}
```

#### E. Content Features (3 features)
```python
features = {
    # 1. Document length (normalized)
    "doc_length": 0.50,

    # 2. Code snippet presence
    "has_code": 1.0,

    # 3. Link/reference count
    "reference_count": 0.35
}
```

### How Features Are Used

```python
# Ranking algorithms consume features in different ways:

# 1. Simple weighted sum
score = 0.5 * semantic_score + 0.3 * bm25_score + 0.2 * recency

# 2. ML model (learns optimal weights)
score = model.predict([
    semantic_score,
    bm25_score,
    exact_match,
    recency,
    thread_depth,
    # ... all 24 features
])

# 3. Rule-based boosting
if exact_match and recency > 0.9:
    score *= 1.5  # Boost recent exact matches
```

---

## How They All Work Together: Complete Example

### Scenario: User searches for "OAuth implementation"

```
┌─────────────────────────────────────────────────────────────────┐
│ STEP 1: SEARCH INFRASTRUCTURE                                   │
├─────────────────────────────────────────────────────────────────┤
│ Semantic Search (ChromaDB):                                     │
│   Query embedding: [0.234, -0.567, ...]                         │
│   Returns: [doc5, doc8, doc12] (similar meaning)                │
│                                                                  │
│ Keyword Search (Elasticsearch + BM25):                          │
│   Query terms: ["oauth", "implementation"]                      │
│   Returns: [doc5, doc12, doc15] (contains keywords)             │
│                                                                  │
│ Candidates: {doc5, doc8, doc12, doc15}                          │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ STEP 2: FEATURE EXTRACTION                                      │
├─────────────────────────────────────────────────────────────────┤
│ For doc5 "Starting OAuth2 implementation today...":             │
│   semantic_score: 0.92      ← High semantic match               │
│   bm25_score: 0.85          ← Contains both keywords            │
│   exact_match: 1.0          ← "OAuth implementation" appears    │
│   recency: 0.95             ← Posted today                      │
│   thread_depth: 0.60        ← Active discussion (8 replies)     │
│   author_authority: 0.75    ← Senior engineer                   │
│   ... (18 more features)                                        │
│                                                                  │
│ For doc8 "authentication setup guide":                          │
│   semantic_score: 0.88      ← Semantically related              │
│   bm25_score: 0.20          ← No keyword match                  │
│   exact_match: 0.0          ← Different terms                   │
│   recency: 0.30             ← 2 weeks old                       │
│   ... (20 more features)                                        │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ STEP 3: RANKING ALGORITHM                                       │
├─────────────────────────────────────────────────────────────────┤
│ Using RRF (Reciprocal Rank Fusion):                             │
│                                                                  │
│ Semantic ranks: {doc5: 1, doc8: 2, doc12: 3}                    │
│ Keyword ranks:  {doc5: 1, doc12: 2, doc15: 3}                   │
│                                                                  │
│ RRF scores:                                                      │
│   doc5:  1/(60+1) + 1/(60+1) = 0.0328  ← Both lists rank 1      │
│   doc12: 1/(60+3) + 1/(60+2) = 0.0320  ← High in both           │
│   doc8:  1/(60+2) + 0        = 0.0161  ← Only semantic          │
│   doc15: 0 + 1/(60+3)        = 0.0159  ← Only keyword           │
│                                                                  │
│ Final Ranking: [doc5, doc12, doc8, doc15]                       │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ STEP 4: EVALUATION METRICS                                      │
├─────────────────────────────────────────────────────────────────┤
│ Ground truth: [doc5: very relevant (3), doc12: relevant (2),    │
│                doc8: somewhat (1), doc15: not relevant (0)]      │
│                                                                  │
│ Our ranking:    [3, 2, 1, 0]                                    │
│ Ideal ranking:  [3, 2, 1, 0] ← Same!                            │
│                                                                  │
│ MRR = 1/1 = 1.0              ✅ First result is relevant        │
│ NDCG@4 = 1.0                 ✅ Perfect ranking                  │
│ Precision@3 = 3/3 = 1.0      ✅ All top 3 are relevant           │
│ Recall@3 = 3/3 = 1.0         ✅ Found all relevant in top 3     │
│                                                                  │
│ Conclusion: Excellent ranking! 🎉                                │
└─────────────────────────────────────────────────────────────────┘
```

---

## Summary Table

| Component | What It Does | When It Runs | Example | Key Question |
|-----------|--------------|--------------|---------|--------------|
| **Search Infrastructure** | Retrieves candidate documents | At query time, first step | ChromaDB: embedding similarity<br>Elasticsearch: BM25 keyword match | "What documents might be relevant?" |
| **Feature Extraction** | Computes signals about relevance | At query time, per candidate | semantic_score: 0.92<br>recency: 0.95<br>thread_depth: 0.60 | "What evidence do we have about this document?" |
| **Ranking Algorithm** | Orders candidates by relevance | At query time, final step | RRF: combine semantic + keyword ranks<br>Weighted: 0.7×semantic + 0.3×keyword<br>ML: learned model | "In what order should we show these?" |
| **Evaluation Metrics** | Measures ranking quality | Offline, during development | MRR: 0.95<br>NDCG@10: 0.87<br>Precision@5: 0.80 | "How good is our ranking?" |

## Key Insights

1. **Search Infrastructure** is about **recall** - finding all potentially relevant items
2. **Feature Extraction** is about **signals** - what evidence helps determine relevance?
3. **Ranking Algorithms** are about **precision** - showing the best items first
4. **Evaluation Metrics** are about **measurement** - quantifying quality improvements

5. **They work sequentially**:
   - Search → Find candidates (100 items)
   - Features → Extract signals (24 features × 100 items)
   - Ranking → Order candidates (top 10)
   - Metrics → Measure quality (MRR, NDCG, etc.)

6. **Optimization happens at different stages**:
   - Tune **search** for better candidate retrieval
   - Add **features** for richer signal
   - Improve **ranking** for better ordering
   - Monitor **metrics** for quality regression

7. **Different components for different goals**:
   - Want to find more relevant docs? → Improve search infrastructure
   - Want better ordering? → Improve ranking algorithm
   - Want new signals? → Add features
   - Want to measure progress? → Track metrics
