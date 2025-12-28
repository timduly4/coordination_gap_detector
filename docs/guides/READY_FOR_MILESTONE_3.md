# Are You Ready for Milestone 3? ✓

## Quick Status Check

### ✅ What's Working Right Now

1. **All Services Running**
   - Elasticsearch ✅
   - Postgres ✅
   - Redis ✅
   - ChromaDB ✅
   - API (FastAPI) ✅

2. **All Tests Passing**
   - Ranking Metrics: 53/53 tests ✅
   - BM25 Scoring: 24/24 tests ✅
   - Hybrid Search: Tests passing ✅
   - Feature Extraction: Tests passing ✅

3. **API Endpoints Available**
   - `/health` ✅
   - `/api/v1/search/` ✅
   - `/api/v1/evaluation/*` ✅
   - `/docs` (OpenAPI) ✅

## 📋 Your 3-Hour Milestone 2 Understanding Plan

### Hour 1: Read the Code (60 min)

**Goal**: Understand how Milestone 2 works

1. **BM25 Implementation** (15 min)
   ```bash
   # Read these sections
   cat src/ranking/scoring.py
   ```

   **Questions to answer**:
   - [ ] What is k1? (term frequency saturation parameter)
   - [ ] What is b? (length normalization parameter)
   - [ ] How is IDF calculated? (log formula)

2. **Hybrid Search** (15 min)
   ```bash
   cat src/search/hybrid_search.py
   ```

   **Questions to answer**:
   - [ ] What is RRF? (Reciprocal Rank Fusion)
   - [ ] What's the RRF formula? (1/(k + rank))
   - [ ] When to use RRF vs weighted? (RRF doesn't need normalization)

3. **Ranking Metrics** (15 min)
   ```bash
   cat src/ranking/metrics.py
   ```

   **Questions to answer**:
   - [ ] What does MRR measure? (first relevant result rank)
   - [ ] What does NDCG measure? (graded relevance with position discount)
   - [ ] What's the DCG formula? ((2^rel - 1) / log2(i + 1))

4. **Feature Engineering** (15 min)
   ```bash
   cat src/ranking/features.py
   ```

   **Questions to answer**:
   - [ ] What feature categories exist? (similarity, temporal, engagement, authority)
   - [ ] How many total features? (~40+)
   - [ ] How are features normalized? (min-max to [0,1])

### Hour 2: Explore Tests (60 min)

**Goal**: See how everything is tested

1. **Read and Run Metric Tests** (20 min)
   ```bash
   # Read a test
   cat tests/test_ranking/test_metrics.py | grep -A 15 "test_mrr_first"

   # Run it
   uv run pytest tests/test_ranking/test_metrics.py::TestMRR -v
   ```

   **Exercise**: Calculate MRR by hand for:
   ```python
   queries = [
       [0, 1, 0],  # First relevant at rank 2 → 1/2 = 0.5
       [1, 0, 0],  # First relevant at rank 1 → 1/1 = 1.0
   ]
   # MRR = (0.5 + 1.0) / 2 = 0.75
   ```

2. **Understand BM25 Tests** (20 min)
   ```bash
   # Read BM25 tests
   cat tests/test_ranking/test_bm25.py | grep -A 20 "test_score_basic"

   # Run with verbose output
   uv run pytest tests/test_ranking/test_bm25.py::TestBM25Scorer::test_score_basic -v -s
   ```

   **Exercise**: What happens when k1 changes?
   ```bash
   uv run pytest tests/test_ranking/test_bm25.py::TestBM25Parameters -v
   ```

3. **Explore Hybrid Search Tests** (20 min)
   ```bash
   # Run hybrid search tests
   uv run pytest tests/test_search/ -v
   ```

### Hour 3: Hands-On Experiments (60 min)

**Goal**: Build intuition by experimenting

1. **BM25 Parameter Tuning** (20 min)

   Create `experiment.py`:
   ```python
   from src.ranking.scoring import BM25Scorer

   # Test k1 effect (term frequency saturation)
   print("Testing k1 (term frequency saturation):")
   for k1 in [0.5, 1.2, 1.5, 2.0]:
       scorer = BM25Scorer(k1=k1, b=0.75)
       score = scorer.score(
           query_terms=["oauth", "implementation"],
           document="implementing OAuth OAuth OAuth implementation",  # Repeated terms
           document_length=4,
           avg_doc_length=10,
           term_idfs={"oauth": 2.0, "implementation": 1.5}
       )
       print(f"  k1={k1}: score={score:.3f}")

   print("\nTesting b (length normalization):")
   # Test b effect (length penalty)
   for b in [0.0, 0.5, 0.75, 1.0]:
       scorer = BM25Scorer(k1=1.5, b=b)

       # Long document
       long_doc = "oauth " * 50  # 50 tokens
       score_long = scorer.score(
           query_terms=["oauth"],
           document=long_doc,
           document_length=50,
           avg_doc_length=10,
           term_idfs={"oauth": 2.0}
       )

       # Short document
       short_doc = "oauth"
       score_short = scorer.score(
           query_terms=["oauth"],
           document=short_doc,
           document_length=1,
           avg_doc_length=10,
           term_idfs={"oauth": 2.0}
       )

       print(f"  b={b}: long={score_long:.3f}, short={score_short:.3f}, ratio={score_short/score_long:.2f}")
   ```

   Run it:
   ```bash
   uv run python experiment.py
   ```

2. **Ranking Metrics Practice** (20 min)

   Create `metrics_practice.py`:
   ```python
   from src.ranking.metrics import calculate_mrr, calculate_ndcg, calculate_precision_at_k

   # Scenario: Search for "OAuth security"
   # You have 5 results with relevance (3=perfect, 2=good, 1=ok, 0=irrelevant)

   scenarios = {
       "Perfect Ranking": [3, 2, 2, 1, 0],
       "Good Ranking": [3, 2, 1, 0, 2],
       "Poor Ranking": [0, 1, 2, 2, 3],
       "Random": [1, 0, 3, 0, 2]
   }

   print("Scenario Comparison:\n")
   for name, relevance in scenarios.items():
       binary = [1 if r > 0 else 0 for r in relevance]

       mrr = calculate_mrr([binary])
       ndcg = calculate_ndcg(relevance, k=5)
       prec = calculate_precision_at_k(binary, k=3)

       print(f"{name:20s} | MRR={mrr:.3f} | NDCG@5={ndcg:.3f} | P@3={prec:.3f}")

   print("\nWhat do you notice?")
   print("- MRR only cares about first relevant item")
   print("- NDCG considers all items and their grades")
   print("- P@3 only looks at top 3 items")
   ```

   ```bash
   uv run python metrics_practice.py
   ```

3. **Hybrid Search Fusion** (20 min)

   Create `fusion_demo.py`:
   ```python
   from src.search.hybrid_search import HybridSearchFusion

   # Simulate semantic and keyword results
   semantic_results = [
       {"id": "doc1", "content": "OAuth security best practices", "semantic_score": 0.95},
       {"id": "doc2", "content": "Authentication guide", "semantic_score": 0.88},
       {"id": "doc3", "content": "API security", "semantic_score": 0.82}
   ]

   keyword_results = [
       {"id": "doc2", "content": "Authentication guide", "keyword_score": 0.92},
       {"id": "doc3", "content": "API security", "keyword_score": 0.88},
       {"id": "doc4", "content": "OAuth specification", "keyword_score": 0.85}
   ]

   print("="*60)
   print("Reciprocal Rank Fusion (RRF)")
   print("="*60)

   rrf_fusion = HybridSearchFusion(strategy="rrf")
   rrf_results = rrf_fusion.fuse(semantic_results, keyword_results)

   for i, result in enumerate(rrf_results, 1):
       details = result['ranking_details']
       print(f"{i}. {result['id']} (score={result['score']:.4f})")
       print(f"   Semantic rank: {details.get('semantic_rank', 'N/A')}")
       print(f"   Keyword rank: {details.get('keyword_rank', 'N/A')}")
       print()

   print("="*60)
   print("Weighted Score Fusion (0.7 semantic, 0.3 keyword)")
   print("="*60)

   weighted_fusion = HybridSearchFusion(strategy="weighted", semantic_weight=0.7, keyword_weight=0.3)
   weighted_results = weighted_fusion.fuse(semantic_results, keyword_results)

   for i, result in enumerate(weighted_results, 1):
       details = result['ranking_details']
       print(f"{i}. {result['id']} (score={result['score']:.4f})")
       print(f"   Semantic: {details.get('semantic_score', 0):.3f}")
       print(f"   Keyword: {details.get('keyword_score', 0):.3f}")
       print()
   ```

   ```bash
   uv run python fusion_demo.py
   ```

## ✅ Readiness Checklist

You're ready for Milestone 3 when you can answer these:

### BM25 Understanding
- [ ] What does k1 control? **Term frequency saturation**
- [ ] What does b control? **Length normalization**
- [ ] Why do common words rank lower? **Higher IDF for rare terms**
- [ ] When would you increase k1? **Technical docs with important repeated terms**

### Hybrid Search Understanding
- [ ] What is RRF formula? **1/(k + rank) where k=60**
- [ ] Why use RRF? **No score normalization needed**
- [ ] When use weighted fusion? **When you trust score distributions**

### Ranking Metrics Understanding
- [ ] What does MRR measure? **Rank of first relevant result**
- [ ] What does NDCG measure? **Position-aware graded relevance**
- [ ] What's range of NDCG? **0 to 1 (1 = perfect)**
- [ ] When is MRR better than NDCG? **Navigational queries (one right answer)**

### Architecture Understanding
- [ ] How does search work? **Semantic (ChromaDB) + Keyword (ES) → Fusion**
- [ ] What's the data flow? **Query → Retrieve → Score → Rank → Return**
- [ ] Where are embeddings? **ChromaDB (vector store)**
- [ ] Where are keyword indices? **Elasticsearch**

## 🚀 How Milestone 2 → Milestone 3

| Milestone 2 Built | Milestone 3 Will Use |
|-------------------|----------------------|
| Semantic search | Find similar discussions (clustering) |
| Feature extraction | Extract entities (people, teams) |
| Ranking algorithms | Rank gap evidence |
| Metrics (MRR, NDCG) | Evaluate gap detection quality |
| BM25 scoring | Score message relevance |

**Key Insight**: Milestone 3 applies search/ranking to a new problem - detecting coordination gaps instead of document retrieval.

## 📚 Resources Created for You

1. **`EXPLORE_MILESTONE_2.md`** - Complete exploration guide (4 levels)
2. **`READY_FOR_MILESTONE_3.md`** - This file (readiness checklist)
3. **`quick_start_milestone2.sh`** - Quick verification script
4. **Passing Tests**: 77+ tests confirming Milestone 2 works

## 🎯 Next Steps (In Order)

1. ✅ **Complete Hour 1**: Read the code (BM25, hybrid, metrics, features)
2. ✅ **Complete Hour 2**: Run and understand tests
3. ✅ **Complete Hour 3**: Run experiments and build intuition
4. ✅ **Answer all checklist questions**
5. ✅ **Read `docs/milestones/MILESTONE_3_BREAKDOWN.md`**
6. ✅ **Start Milestone 3 PR 3A**: Entity Extraction

## 💡 Pro Tips

### When Reading Code
- Start with tests to see expected behavior
- Read method docstrings carefully
- Look for TODOs or FIXMEs
- Check what imports are used

### When Running Tests
- Use `-v` for verbose output
- Use `-s` to see print statements
- Use `-k` to run specific tests: `pytest -k "test_mrr"`
- Use `--pdb` to debug failures

### When Experimenting
- Change one parameter at a time
- Print intermediate results
- Compare before/after
- Try edge cases (empty, very large, negative)

## ⏰ Time Investment

- **Minimum (understand basics)**: 1 hour (read code)
- **Recommended (be confident)**: 3 hours (read + tests + experiments)
- **Deep dive (expert level)**: 6+ hours (include Jupyter notebooks, custom scenarios)

## 🎓 You're Ready When...

You can explain to someone:
- ✅ How BM25 works and why it's better than TF-IDF
- ✅ What RRF does and when to use it
- ✅ The difference between MRR and NDCG
- ✅ How the search pipeline works end-to-end
- ✅ Why Milestone 3 needs these capabilities

---

**Start with Hour 1 right now! Open `src/ranking/scoring.py` and read the BM25Scorer class.**

Good luck! 🚀
