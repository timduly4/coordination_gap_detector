# Coordination Gap Detector - Development Milestones

## Overview
Build this project in stages, with each milestone being a working, demoable version. Each stage adds capability while maintaining a shippable product.

---

## Milestone 1: Foundation & Mock Data (Week 1-2)
**Goal**: Working API with mock data, basic project structure

### Deliverables:
- ✅ Project scaffolding (UV, FastAPI, Docker Compose)
- ✅ Mock data generator for Slack messages
- ✅ Basic API endpoints (`/health`, `/api/v1/search`)
- ✅ Simple semantic search (ChromaDB with mock Slack data)
- ✅ Docker Compose with Postgres, Redis, ChromaDB
- ✅ Basic tests (pytest setup with a few unit tests)
- ✅ README with quickstart instructions

### Success Criteria:
```bash
docker-compose up
curl http://localhost:8000/health
curl http://localhost:8000/api/v1/search -d '{"query": "OAuth"}'
# Returns mock Slack messages with relevance scores
```

### Key Files:
```
src/
  main.py                    # FastAPI app
  config.py                  # Settings
  api/routes/search.py       # Search endpoint
  ingestion/slack/mock_client.py
  db/vector_store.py         # ChromaDB operations
scripts/
  generate_mock_data.py      # Creates realistic scenarios
tests/
  test_search.py
docker-compose.yml
README.md
CLAUDE.md
```

### Time: ~10-15 hours
- Project setup: 2 hours
- Mock data: 3 hours
- Basic search: 4 hours
- Docker setup: 2 hours
- Tests & docs: 4 hours

---

## Milestone 2: Ranking & Search Quality (Week 3-4)
**Goal**: Implement IR fundamentals - BM25, hybrid search, ranking metrics

### Deliverables:
- ✅ Elasticsearch integration for keyword search
- ✅ BM25 scoring implementation
- ✅ Hybrid search (semantic + BM25 fusion)
- ✅ Ranking metrics (MRR, NDCG calculation)
- ✅ Feature engineering for ranking
- ✅ Evaluation script with test queries
- ✅ API returns ranking explanations

### Success Criteria:
```bash
# Search now uses hybrid ranking
curl http://localhost:8000/api/v1/search -d '{
  "query": "OAuth implementation",
  "ranking_strategy": "hybrid"
}'

# Returns results with detailed scoring
{
  "results": [{
    "content": "...",
    "score": 0.89,
    "ranking_features": {
      "semantic_score": 0.92,
      "bm25_score": 0.85,
      "recency": 0.88
    }
  }]
}

# Evaluate ranking quality
python scripts/evaluate_ranking.py
# MRR: 0.72, NDCG@10: 0.68
```

### Key Files:
```
src/
  ranking/
    scoring.py              # BM25 + semantic scoring
    features.py             # Feature extraction
    metrics.py              # MRR, NDCG calculations
  search/
    hybrid_search.py        # Fusion of semantic + keyword
  db/
    elasticsearch.py        # ES client
scripts/
  evaluate_ranking.py       # Offline evaluation
tests/
  test_ranking/
    test_metrics.py
    test_scoring.py
```

### Time: ~12-18 hours
- Elasticsearch setup: 3 hours
- BM25 implementation: 4 hours
- Hybrid search: 3 hours
- Metrics implementation: 4 hours
- Evaluation: 3 hours
- Tests: 3 hours

---

## Milestone 3: Simple Gap Detection (Week 5-6)
**Goal**: Detect one gap type (duplicate work) end-to-end

### Deliverables:
- ✅ Duplicate work detection algorithm
- ✅ Semantic clustering of messages
- ✅ Entity extraction (teams, people)
- ✅ LLM verification with Claude API
- ✅ Gap detection endpoint
- ✅ Impact scoring (simple version)
- ✅ Mock data scenarios for duplicate work

### Success Criteria:
```bash
curl http://localhost:8000/api/v1/gaps/detect -d '{
  "timeframe_days": 30,
  "gap_types": ["duplicate_work"]
}'

# Returns detected gaps with evidence
{
  "gaps": [{
    "type": "DUPLICATE_WORK",
    "title": "Two teams building OAuth integration",
    "teams": ["platform", "auth"],
    "evidence": [...],
    "insight": "Platform team started 4 hours before...",
    "impact_score": 0.89
  }]
}
```

### Key Files:
```
src/
  detection/
    patterns.py             # Base detection interface
    duplicate_work.py       # Duplicate work detector
    impact_scoring.py       # Simple impact score
  analysis/
    entity_extraction.py    # Extract teams, people
  models/
    llm.py                  # Claude API wrapper
    reasoning.py            # LLM verification logic
scripts/
  generate_mock_data.py     # Updated with gap scenarios
tests/
  test_detection/
    test_duplicate_work.py
```

### Time: ~15-20 hours
- Semantic clustering: 4 hours
- Entity extraction: 3 hours
- Detection algorithm: 5 hours
- Claude API integration: 4 hours
- Impact scoring: 2 hours
- Tests & scenarios: 4 hours

---

## Milestone 4: Real Slack Integration (Week 7-8)
**Goal**: Replace mock data with real Slack API integration

### Deliverables:
- ✅ Real Slack OAuth & bot setup
- ✅ Slack webhook receiver
- ✅ Message ingestion (channels, threads)
- ✅ Historical data backfill
- ✅ Real-time event processing
- ✅ Demo Slack workspace setup
- ✅ Configuration for demo vs. real mode

### Success Criteria:
```bash
# Demo mode (default)
docker-compose up
# Uses mock data

# Real Slack mode
DEMO_MODE=false SLACK_BOT_TOKEN=xoxb-... docker-compose up
# Ingests real Slack messages

# Backfill historical data
python scripts/backfill_slack.py --days 90
```

### Key Files:
```
src/
  ingestion/slack/
    client.py               # Real Slack API client
    webhooks.py             # Webhook handlers
    messages.py             # Message processing
    channels.py             # Channel discovery
    threads.py              # Thread context
scripts/
  setup_slack_app.py        # Slack app configuration
  backfill_slack.py         # Historical data import
docs/
  SLACK_SETUP.md            # How to create Slack app
```

### Time: ~12-18 hours
- Slack OAuth: 3 hours
- Message ingestion: 4 hours
- Webhook handling: 3 hours
- Backfill script: 3 hours
- Demo workspace: 2 hours
- Documentation: 3 hours

---

## Milestone 5: Multi-Source Integration (Week 9-11)
**Goal**: Add GitHub and Google Docs ingestion

### Deliverables:
- ✅ GitHub integration (repos, PRs, issues)
- ✅ Google Docs integration (docs, comments)
- ✅ Unified data model across sources
- ✅ Cross-source search
- ✅ Enhanced gap detection with multi-source evidence
- ✅ Mock data for all sources

### Success Criteria:
```bash
# Search across all sources
curl http://localhost:8000/api/v1/search -d '{
  "query": "OAuth implementation",
  "sources": ["slack", "github", "google_docs"]
}'

# Gap detection uses all sources
curl http://localhost:8000/api/v1/gaps/detect
# Evidence includes Slack + GitHub + Docs
```

### Key Files:
```
src/
  ingestion/
    github/
      repos.py
      prs.py
      issues.py
    google/
      docs.py
      drive.py
  search/
    cross_source.py         # Search across sources
  detection/
    duplicate_work.py       # Updated for multi-source
scripts/
  generate_mock_data.py     # All source types
  backfill_github.py
  backfill_google.py
```

### Time: ~20-25 hours
- GitHub integration: 8 hours
- Google Docs integration: 8 hours
- Cross-source search: 4 hours
- Enhanced detection: 4 hours
- Tests & mock data: 5 hours

---

## Milestone 6: Advanced Gap Types (Week 12-13)
**Goal**: Add missing context and stale docs detection

### Deliverables:
- ✅ Missing context detection
- ✅ Stale documentation detection
- ✅ Enhanced impact scoring with multiple signals
- ✅ Gap prioritization algorithm
- ✅ Comprehensive mock scenarios

### Success Criteria:
```bash
# Detect all gap types
curl http://localhost:8000/api/v1/gaps/detect -d '{
  "gap_types": ["duplicate_work", "missing_context", "stale_docs"]
}'

# Returns prioritized gaps across types
# Each with detailed evidence and recommendations
```

### Key Files:
```
src/
  detection/
    missing_context.py
    stale_docs.py
  analysis/
    temporal.py             # Time-based analysis
scripts/
  evaluate_detection.py     # Detection quality metrics
```

### Time: ~15-20 hours
- Missing context: 6 hours
- Stale docs: 6 hours
- Impact scoring: 4 hours
- Evaluation: 4 hours

---

## Milestone 7: ML Ranking Model (Week 14-15)
**Goal**: Train ML model for ranking, implement A/B testing

### Deliverables:
- ✅ Training data generation (click logs)
- ✅ LambdaMART ranking model
- ✅ Feature store with Redis
- ✅ A/B testing framework
- ✅ Model evaluation pipeline
- ✅ Model versioning

### Success Criteria:
```bash
# Train ranking model
python scripts/train_ranking_model.py \
  --training-data data/click_logs.jsonl

# Deploy new model version
python scripts/deploy_model.py --version v2.0

# Run A/B test
curl http://localhost:8000/api/v1/experiments -d '{
  "name": "ml_ranking_v2",
  "treatment": "ml_v2",
  "control": "hybrid_v1"
}'

# Evaluate
python scripts/evaluate_ranking.py --model ml_v2
# MRR improved from 0.72 to 0.78
```

### Key Files:
```
src/
  ranking/
    models.py               # LambdaMART implementation
    experiments.py          # A/B testing framework
  db/
    cache.py                # Feature store
scripts/
  train_ranking_model.py
  evaluate_ranking.py
  deploy_model.py
notebooks/
  ranking_analysis.ipynb
  feature_importance.ipynb
```

### Time: ~18-24 hours
- Training pipeline: 6 hours
- Model implementation: 6 hours
- A/B testing: 4 hours
- Feature store: 3 hours
- Evaluation: 4 hours

---

## Milestone 8: Kubernetes & Production (Week 16-18)
**Goal**: Production-ready deployment with observability

### Deliverables:
- ✅ Kubernetes manifests
- ✅ Terraform infrastructure
- ✅ Prometheus metrics
- ✅ Grafana dashboards
- ✅ Distributed tracing
- ✅ CI/CD pipeline (GitHub Actions)
- ✅ Load testing

### Success Criteria:
```bash
# Deploy to Kubernetes
kubectl apply -f k8s/

# Check metrics
kubectl port-forward svc/prometheus 9090:9090

# Load test
locust -f tests/load/test_api.py --host http://api.example.com

# CI/CD passes
git push origin main
# GitHub Actions: tests pass, builds container, deploys
```

### Key Files:
```
k8s/
  deployment.yaml
  service.yaml
  ingress.yaml
  hpa.yaml                  # Horizontal pod autoscaling
terraform/
  main.tf
  eks.tf                    # EKS cluster
.github/workflows/
  ci.yml
  deploy.yml
src/infrastructure/
  observability.py          # Prometheus metrics
  tracing.py                # Distributed tracing
tests/load/
  test_api.py               # Locust load tests
```

### Time: ~20-25 hours
- K8s manifests: 6 hours
- Terraform: 6 hours
- Observability: 5 hours
- CI/CD: 4 hours
- Load testing: 4 hours

---

## Milestone 9: Web UI (Week 19-20) [Optional]
**Goal**: User-friendly interface for exploring gaps

### Deliverables:
- ✅ React + Next.js frontend
- ✅ Gap dashboard
- ✅ Search interface
- ✅ Evidence viewer
- ✅ Gap management (confirm/dismiss)

### Success Criteria:
- Visit http://localhost:3000
- See dashboard with detected gaps
- Click gap to see evidence
- Search across sources
- Confirm or dismiss gaps

### Key Files:
```
frontend/
  src/
    components/
      GapDashboard.tsx
      SearchInterface.tsx
      EvidenceViewer.tsx
    pages/
      index.tsx
      gaps/[id].tsx
```

### Time: ~25-30 hours
- Project setup: 3 hours
- Dashboard: 8 hours
- Search UI: 6 hours
- Evidence viewer: 6 hours
- Gap management: 6 hours

---

## Total Timeline

**Core Project (Milestones 1-8)**: ~16-18 weeks
- **Minimum viable (1-6)**: ~10-12 weeks
- **Production-ready (1-8)**: ~16-18 weeks
- **With UI (1-9)**: ~19-20 weeks

## Effort Breakdown

| Phase | Hours | Description |
|-------|-------|-------------|
| Foundation | 10-15 | Basic API, mock data, search |
| Ranking | 12-18 | IR fundamentals, metrics |
| Detection | 15-20 | First gap type working |
| Real Slack | 12-18 | Live integration |
| Multi-source | 20-25 | GitHub + Google Docs |
| Advanced Gaps | 15-20 | More detection types |
| ML Ranking | 18-24 | Training, A/B testing |
| Production | 20-25 | K8s, observability |
| **Total Core** | **122-165 hours** | ~3-4 months part-time |

## Recommended Schedule (Part-Time)

**10 hours/week pace:**
- Milestone 1-3: Weeks 1-6 (demoable MVP)
- Milestone 4-6: Weeks 7-12 (real integrations)
- Milestone 7-8: Weeks 13-18 (production-ready)

**20 hours/week pace:**
- Milestone 1-3: Weeks 1-3 (demoable MVP)
- Milestone 4-6: Weeks 4-6 (real integrations)
- Milestone 7-8: Weeks 7-9 (production-ready)

## Git Strategy

Each milestone should have:
```
git checkout -b milestone-1-foundation
# ... work ...
git commit -m "feat: basic search API with mock data"
git push origin milestone-1-foundation
# Create PR, merge to main
git tag v0.1.0
```

Tag conventions:
- v0.1.0 - Milestone 1
- v0.2.0 - Milestone 2
- v0.3.0 - Milestone 3
- v1.0.0 - Milestone 8 (production-ready)

## Each Milestone Should Be:

✅ **Demoable**: Can show someone working functionality
✅ **Tested**: Tests pass, coverage documented
✅ **Documented**: README updated, CHANGELOG entry
✅ **Deployable**: docker-compose up works
✅ **Mergeable**: PR ready, CI passes

## Tips for Success

1. **Don't skip milestones** - each builds on previous
2. **Commit frequently** - show real development progression
3. **Write tests as you go** - don't save for later
4. **Document immediately** - future you will forget context
5. **Use Claude Code** - accelerate development with AI
6. **Get each milestone working** before moving on
7. **It's OK to simplify** - perfect is the enemy of done

## Interview-Ready Checkpoints

- **After Milestone 3**: Can demo gap detection end-to-end
- **After Milestone 5**: Can show multi-source integration
- **After Milestone 7**: Can discuss ML ranking in depth
- **After Milestone 8**: Can discuss production architecture

You can interview confidently after Milestone 3-5 with a working, demoable system.