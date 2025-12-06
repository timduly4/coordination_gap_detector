# CLAUDE.md

## Project Overview
This is an AI-powered coordination gap detection system built with Python, demonstrating expertise in information retrieval, ranking algorithms, and distributed systems architecture. The project identifies and resolves coordination failures across enterprise communication channels - helping organizations detect when teams are duplicating work, missing critical context, or working at cross-purposes.

## What This Project Does
An enterprise platform that ingests data from Slack, Google Docs, GitHub, and other collaboration tools to automatically identify coordination gaps. The system uses advanced search quality techniques, ML-based ranking, and LLM reasoning to surface issues like:
- Multiple teams solving the same problem independently
- Critical decisions made without stakeholder awareness
- Outdated documentation contradicting current implementations
- Knowledge silos preventing effective collaboration
- Missing context that could prevent costly mistakes

## Tech Stack

### Core Framework
- **Python 3.11+** - Base language
- **UV** - Fast dependency management and project setup
- **FastAPI** - Async web framework for REST API
- **Pydantic** - Data validation and settings management

### AI/ML Components
- **Anthropic Claude API** - Primary LLM for reasoning about coordination patterns
- **Claude Code** - AI-native development workflow integration
- **LangChain** - LLM application framework for multi-step reasoning
- **ChromaDB** - Vector database for semantic search across documents
- **scikit-learn** - ML-based ranking models and pattern detection
- **tiktoken** - Token counting and text chunking

### Search & Ranking
- **Elasticsearch** - Full-text search across all data sources
- **Custom Ranking Pipeline** - Hybrid scoring with context relevance signals
- **A/B Testing Framework** - Online experimentation for detection improvements
- **Metrics Tracking** - MRR, NDCG@k, DCG, precision/recall for gap detection

### Data Ingestion & Integration
- **Slack SDK** - Real-time message ingestion, channel analysis
- **Google Workspace APIs** - Docs, Sheets, Drive content
- **GitHub API** - Code, PRs, issues, discussions
- **Jira API** - Project tracking and work item analysis
- **Confluence API** - Documentation and wiki content
- **Webhook receivers** - Real-time event processing

### Data & Storage
- **PostgreSQL with pgvector** - Persistent storage, vector search, graph queries
- **Redis** - Caching, real-time feature store, event queue
- **SQLAlchemy** - ORM for database operations
- **Neo4j (optional)** - Knowledge graph for org relationships

### Infrastructure & Orchestration
- **Kubernetes** - Container orchestration and deployment
- **Docker & Docker Compose** - Containerization
- **Terraform** - Infrastructure as code
- **Prometheus + Grafana** - Metrics and monitoring
- **ArgoCD** - GitOps continuous deployment
- **Kafka** - Event streaming for high-volume ingestion

### Development Tools
- **pytest** - Testing framework with extensive fixtures
- **GitHub Actions** - CI/CD pipeline
- **Claude Code** - AI-assisted development

## Project Structure

```
coordination-gap-detector/
├── pyproject.toml              # UV dependencies
├── .env.example                # Environment variable template
├── docker-compose.yml          # Local development setup
├── Dockerfile                  # Application container
├── k8s/                        # Kubernetes manifests
│   ├── deployment.yaml
│   ├── service.yaml
│   ├── ingress.yaml
│   ├── kafka.yaml
│   └── configmap.yaml
├── terraform/                  # Infrastructure as code
│   ├── main.tf
│   ├── variables.tf
│   └── outputs.tf
├── README.md                   # User-facing documentation
├── CLAUDE.md                   # This file - AI context
│
├── src/
│   ├── __init__.py
│   ├── main.py                 # FastAPI application entry
│   ├── config.py               # Settings and environment config
│   │
│   ├── api/
│   │   ├── __init__.py
│   │   ├── routes/
│   │   │   ├── gaps.py         # Coordination gap endpoints
│   │   │   ├── search.py       # Cross-source search
│   │   │   ├── insights.py     # AI-generated insights
│   │   │   ├── metrics.py      # Detection quality metrics
│   │   │   └── health.py       # Health checks
│   │   └── dependencies.py     # FastAPI dependencies
│   │
│   ├── detection/              # Gap detection engine
│   │   ├── __init__.py
│   │   ├── patterns.py         # Coordination anti-patterns
│   │   ├── duplicate_work.py   # Detect parallel efforts
│   │   ├── missing_context.py  # Identify information gaps
│   │   ├── stale_docs.py       # Documentation drift detection
│   │   ├── knowledge_silo.py   # Team isolation patterns
│   │   └── impact_scoring.py   # Prioritize gaps by impact
│   │
│   ├── ranking/                # Search quality & ranking
│   │   ├── __init__.py
│   │   ├── models.py           # ML ranking models
│   │   ├── features.py         # Feature engineering for ranking
│   │   ├── scoring.py          # Relevance scoring strategies
│   │   ├── reranker.py         # Cross-encoder reranking
│   │   ├── metrics.py          # MRR, NDCG, DCG calculations
│   │   └── experiments.py      # A/B testing framework
│   │
│   ├── search/
│   │   ├── __init__.py
│   │   ├── query_parser.py     # Query understanding & expansion
│   │   ├── retrieval.py        # Multi-stage retrieval
│   │   ├── hybrid_search.py    # Semantic + keyword fusion
│   │   ├── cross_source.py     # Search across Slack/Docs/GitHub
│   │   └── filters.py          # Dynamic filtering logic
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── embeddings.py       # Embedding generation
│   │   ├── llm.py              # LLM interaction wrapper
│   │   ├── reasoning.py        # Multi-step LLM reasoning
│   │   └── schemas.py          # Pydantic models
│   │
│   ├── ingestion/              # Data source ingestion
│   │   ├── __init__.py
│   │   ├── base.py             # Base ingestion interface
│   │   ├── slack/
│   │   │   ├── __init__.py
│   │   │   ├── client.py       # Slack API client
│   │   │   ├── messages.py     # Message processing
│   │   │   ├── channels.py     # Channel analysis
│   │   │   └── threads.py      # Thread context extraction
│   │   ├── google/
│   │   │   ├── __init__.py
│   │   │   ├── docs.py         # Google Docs ingestion
│   │   │   ├── sheets.py       # Sheets analysis
│   │   │   └── drive.py        # Drive file discovery
│   │   ├── github/
│   │   │   ├── __init__.py
│   │   │   ├── repos.py        # Repository analysis
│   │   │   ├── prs.py          # Pull request tracking
│   │   │   ├── issues.py       # Issue discussion analysis
│   │   │   └── commits.py      # Commit history parsing
│   │   └── webhooks.py         # Real-time webhook handlers
│   │
│   ├── analysis/               # Content analysis
│   │   ├── __init__.py
│   │   ├── entity_extraction.py # People, teams, projects
│   │   ├── topic_modeling.py    # Discussion topic clustering
│   │   ├── sentiment.py         # Sentiment and urgency
│   │   ├── temporal.py          # Time-based patterns
│   │   └── relationships.py     # Org graph construction
│   │
│   ├── db/
│   │   ├── __init__.py
│   │   ├── vector_store.py     # ChromaDB + pgvector operations
│   │   ├── elasticsearch.py    # ES client and indexing
│   │   ├── postgres.py         # PostgreSQL operations
│   │   ├── graph.py            # Neo4j graph queries
│   │   └── cache.py            # Redis caching
│   │
│   ├── infrastructure/
│   │   ├── __init__.py
│   │   ├── observability.py    # Prometheus metrics
│   │   ├── rate_limiting.py    # Distributed rate limiter
│   │   ├── circuit_breaker.py  # Resilience patterns
│   │   └── streaming.py        # Kafka event processing
│   │
│   └── utils/
│       ├── __init__.py
│       ├── text_processing.py  # NLP utilities
│       ├── time_utils.py       # Temporal analysis helpers
│       └── logging.py          # Structured logging
│
├── tests/
│   ├── __init__.py
│   ├── conftest.py             # Pytest fixtures
│   ├── test_detection/         # Gap detection tests
│   │   ├── test_duplicate_work.py
│   │   ├── test_missing_context.py
│   │   └── test_impact_scoring.py
│   ├── test_ranking/           # Ranking algorithm tests
│   │   ├── test_metrics.py     # MRR, NDCG validation
│   │   └── test_scoring.py     # Scoring strategy tests
│   ├── test_ingestion/         # Data source tests
│   ├── test_search/
│   └── test_integration/       # End-to-end tests
│
├── notebooks/
│   ├── gap_analysis.ipynb      # Gap pattern exploration
│   ├── ranking_eval.ipynb      # Offline ranking evaluation
│   ├── org_graph.ipynb         # Organization network analysis
│   └── ab_test_results.ipynb   # Experiment analysis
│
├── scripts/
│   ├── setup.sh                # Initial setup script
│   ├── seed_data.sh            # Load sample data
│   ├── evaluate_detection.py   # Offline detection metrics
│   ├── backfill_sources.py     # Historical data ingestion
│   └── deploy.sh               # Deployment automation
│
└── frontend/                   # Optional web UI
    ├── package.json
    ├── src/
    │   ├── components/
    │   │   ├── GapDashboard.tsx
    │   │   ├── SearchInterface.tsx
    │   │   └── InsightCard.tsx
    │   └── pages/
```

## Key Design Decisions

### Coordination Gap Detection Architecture

**What is a Coordination Gap?**

A coordination gap occurs when organizational information flows break down, causing:
- **Duplicate Work**: Two teams building the same feature independently
- **Missing Context**: Decisions made without critical stakeholder input
- **Stale Documentation**: Docs contradicting current implementation
- **Knowledge Silos**: Critical knowledge trapped in individual teams
- **Missed Dependencies**: Work proceeding unaware of blocking issues

**Multi-Signal Detection Pipeline:**

1. **Data Ingestion** - Stream events from all sources (Slack, Docs, GitHub)
2. **Entity Extraction** - Identify people, teams, projects, topics
3. **Semantic Indexing** - Embed content for similarity search
4. **Pattern Detection** - Apply gap detection algorithms
5. **Impact Scoring** - Rank gaps by potential organizational cost
6. **LLM Reasoning** - Generate actionable insights with Claude
7. **Alert Routing** - Notify relevant stakeholders

**Example Gap Detection: Duplicate Work**

```python
# Simplified algorithm
def detect_duplicate_work(timeframe_days=30):
    # 1. Extract all technical discussions
    discussions = get_discussions(sources=['slack', 'github', 'docs'])
    
    # 2. Cluster by semantic similarity
    clusters = semantic_clustering(discussions, threshold=0.85)
    
    # 3. Identify clusters with multiple teams
    for cluster in clusters:
        teams = extract_teams(cluster.discussions)
        if len(teams) > 1:
            # 4. Check for temporal overlap (working simultaneously)
            if has_temporal_overlap(cluster.discussions):
                # 5. Verify they're actually solving same problem
                if llm_verify_duplicate(cluster):
                    gap = CoordinationGap(
                        type='DUPLICATE_WORK',
                        teams=teams,
                        evidence=cluster.discussions,
                        impact_score=calculate_impact(cluster),
                        recommendation=llm_generate_recommendation(cluster)
                    )
                    yield gap
```

**Impact Scoring Features:**
- Team size and seniority involved
- Engineering time invested (commit volume, discussion length)
- Project criticality (tied to OKRs, roadmap items)
- Historical cost of similar gaps
- Velocity impact (blocking other work)

### Search Quality & Ranking Architecture

**Why Ranking Matters for Gap Detection:**

When a potential gap is detected, the system must:
1. Retrieve all related discussions across sources
2. Rank them by relevance to the gap pattern
3. Surface the most important evidence first
4. Enable users to verify or dismiss the gap

**Multi-Stage Retrieval Pipeline:**
1. **Stage 1: Candidate Retrieval** - Hybrid search (semantic + BM25) across sources
2. **Stage 2: Cross-Source Fusion** - Merge results from Slack, Docs, GitHub
3. **Stage 3: Feature Extraction** - Compute ranking signals (recency, authority, engagement)
4. **Stage 4: ML Ranking** - LambdaMART model trained on verified gaps
5. **Stage 5: Reranking** - Claude-based relevance verification
6. **Stage 6: Evidence Chain** - Build causal chain of related items

**Ranking Features (40+ signals):**
- Query-document similarity (cosine, BM25 score)
- Source authority (team influence, doc ownership)
- Temporal signals (recency, activity burst)
- Engagement metrics (thread depth, participant count)
- Cross-source consistency (same topic across channels)
- Entity overlap (same people/teams involved)
- Topic relevance to detected gap type

**Evaluation Metrics:**
- **MRR (Mean Reciprocal Rank)** - Primary metric for top relevant item
- **NDCG@10** - Graded relevance for evidence ranking
- **DCG** - Cumulative gain with position discount
- **Precision@k / Recall@k** - Coverage of true gap evidence
- **Gap Verification Rate** - % of detected gaps confirmed by users

### Information Retrieval Fundamentals

**IDF (Inverse Document Frequency):**
- Weights terms by rarity across corpus
- Common organizational terms (e.g., "meeting", "update") get lower weight
- Specific project/technical terms get higher weight
- Used in BM25 scoring for keyword matching

**BM25 Scoring:**
```python
# Probabilistic ranking function
def bm25_score(query_terms, document, k1=1.5, b=0.75):
    score = 0
    for term in query_terms:
        idf = calculate_idf(term)
        tf = term_frequency(term, document)
        doc_len = len(document)
        avg_doc_len = corpus_average_length()
        
        numerator = tf * (k1 + 1)
        denominator = tf + k1 * (1 - b + b * (doc_len / avg_doc_len))
        score += idf * (numerator / denominator)
    return score
```

**Semantic Similarity:**
- Dense vector embeddings via Claude API
- Handles synonyms and paraphrasing
- Cross-source semantic matching (Slack → GitHub)
- Temporal drift detection (old doc vs new code)

### Why This Architecture?

**Multi-Source Integration:**
- Organizations communicate across 5-10+ tools
- Gaps emerge from cross-tool blindspots
- Single-source analysis misses coordination failures
- Real-time ingestion catches gaps as they form

**ML-Based Gap Detection:**
- Hand-coded rules miss novel gap patterns
- Learns from historical verified gaps
- Adapts to organizational communication patterns
- Improves with user feedback (confirmed/dismissed)

**LLM Reasoning Layer:**
- Explains WHY something is a gap
- Generates actionable recommendations
- Verifies semantic similarity beyond embeddings
- Produces human-readable insights

**Distributed Systems Design:**
- Kubernetes for horizontal scaling (thousands of users)
- Event streaming with Kafka (high-volume real-time data)
- Circuit breakers for external API resilience
- Distributed caching with Redis
- Async processing for heavy analysis workloads

### Why Claude API & Claude Code?
- Exceptional reasoning about organizational context
- Large context window for analyzing long threads
- Structured output for gap verification
- Reliable explanation generation
- Claude Code integration for AI-native development

### Infrastructure & Kubernetes Strategy
- Multi-region deployment for global enterprises
- Blue-green deployments for zero downtime
- Horizontal pod autoscaling based on ingestion volume
- Kafka for decoupled event processing
- Observability with Prometheus + distributed tracing

## Environment Variables

```bash
# API Keys
ANTHROPIC_API_KEY=your_key_here
SLACK_BOT_TOKEN=xoxb-your-token
SLACK_APP_TOKEN=xapp-your-token
GOOGLE_CREDENTIALS_JSON=path/to/creds.json
GITHUB_TOKEN=your_github_token
JIRA_API_TOKEN=your_jira_token

# Search Infrastructure
ELASTICSEARCH_URL=https://elasticsearch:9200
ELASTICSEARCH_API_KEY=your_es_key

# Database
POSTGRES_URL=postgresql://user:pass@localhost:5432/coordination
REDIS_URL=redis://localhost:6379
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password

# Vector Store
CHROMA_PERSIST_DIR=./data/chroma

# Event Streaming
KAFKA_BOOTSTRAP_SERVERS=localhost:9092
KAFKA_TOPIC_SLACK=slack-events
KAFKA_TOPIC_GITHUB=github-events

# Kubernetes (Production)
K8S_NAMESPACE=coordination-prod
K8S_CONTEXT=production-cluster

# Observability
PROMETHEUS_ENDPOINT=http://prometheus:9090
GRAFANA_API_KEY=your_grafana_key
SENTRY_DSN=your_sentry_dsn

# Application
API_HOST=0.0.0.0
API_PORT=8000
LOG_LEVEL=INFO
ENVIRONMENT=production
MAX_WORKERS=8

# Feature Flags
ENABLE_REALTIME_DETECTION=true
ENABLE_DUPLICATE_WORK_DETECTION=true
ENABLE_MISSING_CONTEXT_DETECTION=true
ENABLE_STALE_DOCS_DETECTION=true
GAP_DETECTION_MODEL_VERSION=v3.1
RANKING_MODEL_VERSION=v2.7
```

## Getting Started

### Local Development with UV
```bash
# Install UV
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and setup
git clone <repo>
cd coordination-gap-detector
uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"

# Environment setup
cp .env.example .env
# Edit .env with your API keys and source credentials

# Start infrastructure (Postgres, Redis, Elasticsearch, Kafka)
docker-compose up -d

# Run migrations
uv run alembic upgrade head

# Start background workers (ingestion, detection)
uv run celery -A src.workers worker --loglevel=info &

# Start development server
uv run uvicorn src.main:app --reload --port 8000
```

### Initial Data Ingestion
```bash
# Backfill historical data (last 90 days)
uv run python scripts/backfill_sources.py \
  --sources slack,github,google_docs \
  --days 90

# Set up real-time webhooks
uv run python scripts/setup_webhooks.py
```

### Kubernetes Deployment
```bash
# Build and push image
docker build -t coordination-detector:latest .
docker push your-registry/coordination-detector:latest

# Deploy with Terraform
cd terraform
terraform init
terraform plan
terraform apply

# Or apply k8s manifests directly
kubectl apply -f k8s/

# Deploy via ArgoCD (GitOps)
argocd app create coordination \
  --repo https://github.com/you/coordination-gap-detector \
  --path k8s \
  --dest-server https://kubernetes.default.svc \
  --dest-namespace coordination
```

## API Endpoints

### Detect Coordination Gaps
```bash
POST /api/v1/gaps/detect
{
  "timeframe_days": 30,
  "sources": ["slack", "github", "google_docs"],
  "gap_types": ["duplicate_work", "missing_context", "stale_docs"],
  "teams": ["engineering", "product"],
  "min_impact_score": 0.7
}

Response:
{
  "gaps": [
    {
      "id": "gap_abc123",
      "type": "DUPLICATE_WORK",
      "title": "Two teams building OAuth integration simultaneously",
      "impact_score": 0.89,
      "teams_involved": ["platform-team", "auth-team"],
      "evidence": [
        {
          "source": "slack",
          "channel": "#platform",
          "message": "Starting OAuth2 implementation...",
          "timestamp": "2024-12-01T10:30:00Z",
          "author": "alice@company.com"
        },
        {
          "source": "github",
          "repo": "auth-service",
          "pr": "#245",
          "title": "Add OAuth2 support",
          "timestamp": "2024-12-01T14:20:00Z",
          "author": "bob@company.com"
        }
      ],
      "insight": "Platform and Auth teams are independently implementing OAuth2. Platform team started in Slack 4 hours before Auth opened PR. High overlap in scope - consider consolidating efforts.",
      "recommendation": "Connect alice@company.com and bob@company.com. Consider having one team lead with the other contributing specific components.",
      "estimated_cost": "~40 engineering hours of duplicate effort"
    }
  ],
  "metadata": {
    "total_gaps_detected": 1,
    "detection_time_ms": 3200,
    "model_version": "v3.1"
  }
}
```

### Search Across Sources
```bash
POST /api/v1/search
{
  "query": "OAuth implementation decisions",
  "sources": ["slack", "github", "google_docs"],
  "date_range": {
    "start": "2024-11-01",
    "end": "2024-12-01"
  },
  "ranking_strategy": "ml_hybrid"
}

Response:
{
  "results": [
    {
      "source": "slack",
      "content": "We decided to use Auth0 for OAuth...",
      "channel": "#architecture",
      "timestamp": "2024-11-15T09:00:00Z",
      "score": 0.92,
      "ranking_features": {
        "semantic_score": 0.94,
        "bm25_score": 0.88,
        "recency": 0.95,
        "authority": 0.91
      }
    }
  ]
}
```

### Gap Metrics & Quality
```bash
GET /api/v1/metrics/detection-quality
{
  "metrics": {
    "mrr": 0.74,
    "ndcg_at_10": 0.79,
    "gap_verification_rate": 0.68,
    "false_positive_rate": 0.15
  },
  "by_gap_type": {
    "duplicate_work": {
      "precision": 0.82,
      "recall": 0.71
    },
    "missing_context": {
      "precision": 0.65,
      "recall": 0.58
    }
  }
}
```

## Development Workflow

### AI-Native Development with Claude Code
```bash
# Use Claude Code for development tasks
claude-code "implement missing context detection algorithm"
claude-code "add Jira integration for project tracking"
claude-code "optimize gap detection pipeline for latency"

# Code review with AI assistance
git diff | claude-code "review this gap detection change"
```

### Testing Strategy

**Unit Tests:**
```bash
# Test gap detection algorithms
pytest tests/test_detection/ -v

# Test ranking metrics
pytest tests/test_ranking/test_metrics.py
```

**Integration Tests:**
```bash
# End-to-end gap detection
pytest tests/test_integration/test_gap_pipeline.py

# Test with real data sources (mocked)
pytest tests/test_integration/ --use-mock-sources
```

**Detection Quality Evaluation:**
```bash
# Offline evaluation with labeled gaps
python scripts/evaluate_detection.py \
  --model v3.1 \
  --test-set data/labeled_gaps.jsonl \
  --metrics precision,recall,mrr,ndcg

# Compare detection strategies
python scripts/evaluate_detection.py --compare \
  --models rule_based,ml_hybrid,llm_only
```

## Real-World Coordination Gaps Examples

### 1. Duplicate Work Detection
**Signal Pattern:**
- Two Slack channels discussing same feature
- Parallel GitHub PRs with overlapping functionality
- Similar Google Docs technical specs
- Temporal overlap (same timeframe)
- No cross-references between teams

**Detection Algorithm:**
- Semantic clustering of technical discussions
- Entity extraction (features, components)
- Team membership analysis
- Temporal co-occurrence check
- LLM verification of true duplication

### 2. Missing Context Detection
**Signal Pattern:**
- Important decision in Slack without key stakeholders
- GitHub PR merged without required reviewers
- Google Doc finalized without security review
- Jira epic started without dependency check

**Detection Algorithm:**
- Extract decision points (keywords: "decided", "going with", "approved")
- Identify required stakeholders (org chart, RACI matrix)
- Check participant list against required list
- Score impact of missing perspective
- Generate catch-up summary for missing parties

### 3. Stale Documentation Detection
**Signal Pattern:**
- Google Doc describes process contradicted by recent code
- Confluence page unchanged while GitHub shows major refactor
- Onboarding docs reference deprecated systems
- API docs mismatch current endpoint behavior

**Detection Algorithm:**
- Extract implementation details from docs
- Compare with current codebase state
- Detect semantic drift over time
- Score staleness by edit gap and code changes
- Generate doc update recommendations

### 4. Knowledge Silo Detection
**Signal Pattern:**
- Critical knowledge only in one team's Slack channel
- Single point of failure (one person with expertise)
- No cross-team discussions on shared dependencies
- Documentation exists but not discoverable

**Detection Algorithm:**
- Build knowledge graph (who knows what)
- Identify critical knowledge (high importance, low redundancy)
- Detect single points of failure
- Measure cross-team information flow
- Recommend knowledge sharing actions

## Performance Characteristics

### Latency Targets
- **Real-time ingestion**: <100ms event processing (p95)
- **Gap detection**: <5s for single gap type (p95)
- **Full scan**: <30s across all sources (p95)
- **Search query**: <200ms (p95)

### Scale
- **Data sources**: 10+ integrations per organization
- **Events per day**: 100K+ (Slack, GitHub, etc.)
- **Indexed documents**: 10M+ across sources
- **Concurrent users**: 10,000+
- **Organizations**: 1,000+ supported

### Optimization Strategies
- Event streaming with Kafka (decouple ingestion from processing)
- Incremental detection (process only new/changed data)
- Cached embeddings (24h TTL)
- Batch LLM calls (analyze multiple gaps together)
- Precomputed org graphs (hourly refresh)
- Distributed search (sharded Elasticsearch)

## Monitoring & Observability

### Key Metrics
- **Detection Quality**: Precision, recall, MRR, NDCG, verification rate
- **Performance**: Event processing lag, detection latency, search latency
- **Engagement**: Gaps reviewed, confirmed, dismissed, acted upon
- **System Health**: Ingestion throughput, error rate, API quota usage

### Dashboards
- Gap detection quality trends
- Source-specific ingestion health
- Organization network topology
- User engagement with detected gaps
- Cost savings from prevented duplicate work

### Alerting
- Detection precision drops below 0.60 (critical)
- Event processing lag > 5 minutes (warning)
- API rate limit near exhaustion (warning)
- Critical gap detected (high impact score) (info)

## Key Technical Capabilities Demonstrated

This project showcases expertise in modern AI and distributed systems:

✅ **AI-Powered Enterprise Collaboration**: Production-ready coordination improvement system
✅ **Intelligent Gap Identification**: Automated detection of organizational inefficiencies
✅ **Search Quality Expertise**: Advanced IR concepts (MRR, NDCG, IDF, BM25)
✅ **Ranking Models**: ML-based scoring, feature engineering, evaluation
✅ **A/B Testing**: Online experimentation framework with statistical rigor
✅ **Distributed Systems**: Kubernetes, Kafka, scalability, reliability patterns
✅ **Backend Focus**: FastAPI, async processing, event streaming, API design
✅ **Multi-Source Integration**: Slack, Google Docs, GitHub integration patterns
✅ **AI-Native Development**: Claude Code integration, LLM reasoning
✅ **Enterprise Scale**: Multi-tenant architecture, thousands of users
✅ **Hands-On Implementation**: Complete working system with production considerations
✅ **First Principles Thinking**: Justified design decisions with explicit tradeoffs
✅ **End-to-End Ownership**: Research, design, implementation, and deployment
✅ **Real-World Impact**: Measurable cost savings and efficiency improvements

## Future Enhancements

- [ ] Predictive gap detection (prevent before they occur)
- [ ] Auto-remediation (suggest concrete actions, not just alerts)
- [ ] Organizational health score (aggregate coordination quality)
- [ ] Custom gap patterns per organization
- [ ] Integration with project management tools (Asana, Linear)
- [ ] Browser extension for inline gap warnings
- [ ] Mobile app for executive dashboards
- [ ] GraphRAG for deep organizational knowledge
- [ ] Fine-tuned models for specific industries
- [ ] Privacy-preserving analysis for sensitive data

## Useful Commands

```bash
# Gap detection evaluation
python scripts/evaluate_detection.py --model current --test-set labeled

# Backfill historical data
python scripts/backfill_sources.py --source slack --days 90

# Deploy to Kubernetes
kubectl apply -f k8s/

# Scale ingestion workers
kubectl scale deployment/ingestion-worker --replicas=20

# Monitor event lag
kubectl logs -f deployment/kafka-consumer | grep lag

# Claude Code assisted development
claude-code "add detection algorithm for knowledge silos"
claude-code "optimize Slack ingestion for 100k messages/day"
```

## Resources

- [Information Retrieval - Manning](https://nlp.stanford.edu/IR-book/)
- [Learning to Rank Guide](https://opensourceconnections.com/blog/2017/02/24/what-is-learning-to-rank/)
- [Kafka: The Definitive Guide](https://www.confluent.io/resources/kafka-the-definitive-guide/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Kubernetes Best Practices](https://kubernetes.io/docs/concepts/)
- [Slack API Documentation](https://api.slack.com/)
- [Google Workspace APIs](https://developers.google.com/workspace)
- [Claude API Reference](https://docs.anthropic.com/)
- [Claude Code Documentation](https://docs.anthropic.com/claude/docs/claude-code)

---

**Last Updated**: December 2024  
**Python Version**: 3.11+  
**Kubernetes**: 1.28+  
**License**: MIT