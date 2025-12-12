# Coordination Gap Detector

AI-powered coordination gap detection system for identifying failures across enterprise communication channels. This platform automatically detects when teams are duplicating work, missing critical context, or working at cross-purposes across Slack, GitHub, Google Docs, and other collaboration tools.

## What This Project Does

An enterprise platform that ingests data from multiple collaboration tools to automatically identify coordination gaps including:

- **Duplicate Work** - Multiple teams solving the same problem independently
- **Missing Context** - Critical decisions made without stakeholder awareness
- **Stale Documentation** - Outdated docs contradicting current implementations
- **Knowledge Silos** - Critical knowledge trapped in individual teams
- **Missed Dependencies** - Work proceeding unaware of blocking issues

## Key Features

- **Multi-Source Integration** - Slack, GitHub, Google Docs, Jira, Confluence
- **Real-Time Detection** - Identify coordination gaps as they emerge
- **ML-Based Ranking** - Prioritize gaps by organizational impact
- **LLM Reasoning** - Claude-powered insights and recommendations
- **Search Quality** - Advanced IR with MRR, NDCG evaluation metrics
- **Distributed Architecture** - Kubernetes, Kafka, Elasticsearch for scale

## Tech Stack

- **Python 3.11+** with FastAPI
- **UV** for dependency management
- **Claude API** for LLM reasoning
- **ChromaDB** for vector search
- **Elasticsearch** for full-text search
- **PostgreSQL** with pgvector
- **Redis** for caching
- **Kafka** for event streaming
- **Kubernetes** for orchestration

## Quick Start

### Prerequisites

- Python 3.11 or higher
- UV package manager ([installation guide](https://github.com/astral-sh/uv))
- Docker and Docker Compose (for infrastructure)

### Installation

```bash
# Install UV (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone the repository
git clone https://github.com/timduly4/coordination_gap_detector.git
cd coordination_gap_detector

# Create virtual environment and install dependencies
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -e ".[dev]"

# Copy environment template and configure
cp .env.example .env
# Edit .env with your API keys and configuration
```

### Start Infrastructure

```bash
# Start required services (PostgreSQL, Redis, ChromaDB, Redis)
docker compose up -d

# Run database migrations
docker compose exec api alembic upgrade head
```

### Generate Mock Data (Development)

For development and testing, you can generate realistic Slack conversation data:

```bash
# Generate all mock conversation scenarios
docker compose exec api python scripts/generate_mock_data.py --scenarios all

# Generate specific scenarios
docker compose exec api python scripts/generate_mock_data.py --scenarios oauth_discussion decision_making

# Clear existing data before generating
docker compose exec api python scripts/generate_mock_data.py --scenarios all --clear
```

**Available Mock Scenarios:**
- `oauth_discussion` - OAuth implementation discussion showing potential duplicate work (8 messages)
- `decision_making` - Team decision made without key stakeholders (5 messages)
- `bug_report` - Bug report and resolution workflow (5 messages)
- `feature_planning` - Cross-team feature planning coordination (6 messages)

**Verify Data:**
```bash
# Check message count in database
docker compose exec postgres psql -U coordination_user -d coordination -c "SELECT COUNT(*) FROM messages;"

# View messages by channel
docker compose exec postgres psql -U coordination_user -d coordination -c "SELECT channel, COUNT(*) FROM messages GROUP BY channel;"
```

### Run the Application

```bash
# Development mode with auto-reload
uv run uvicorn src.main:app --reload --port 8000

# Or use the main module directly
uv run python -m src.main
```

The API will be available at:
- **API**: http://localhost:8000
- **Interactive Docs**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

## Development

### Project Structure

```
coordination-gap-detector/
├── src/                    # Main application code
│   ├── api/               # FastAPI routes and endpoints
│   │   └── routes/        # API route modules (search, gaps, etc.)
│   ├── services/          # Business logic layer
│   ├── detection/         # Gap detection algorithms
│   ├── ranking/           # Search quality & ranking
│   ├── search/            # Multi-source search
│   ├── models/            # Data models and schemas
│   ├── ingestion/         # Data source integrations
│   ├── analysis/          # Content analysis
│   ├── db/                # Database operations
│   ├── infrastructure/    # Observability, rate limiting
│   └── utils/             # Utilities
├── tests/                 # Test suite
│   ├── test_api/          # API endpoint tests
│   ├── test_detection/    # Detection algorithm tests
│   └── test_integration/  # Integration tests
├── notebooks/             # Jupyter notebooks for analysis
├── scripts/               # Utility scripts
├── k8s/                   # Kubernetes manifests
└── terraform/             # Infrastructure as code
```

### Running Tests

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=src --cov-report=html

# Run specific test modules
uv run pytest tests/test_api/test_search.py -v          # Search API tests
uv run pytest tests/test_vector_store.py -v             # Vector store tests
uv run pytest tests/test_embeddings.py -v               # Embedding tests

# Run tests by category
uv run pytest tests/test_api/ -v                        # All API tests
uv run pytest tests/test_integration/ -v                # Integration tests

# Run in Docker
docker compose exec api pytest

# Run with verbose output
docker compose exec api pytest -v

# Run specific test with Docker
docker compose exec api pytest tests/test_api/test_search.py::TestSearchEndpoint::test_search_basic_query -v
```

For comprehensive testing documentation including test structure, options, and troubleshooting, see **[TESTING.md](./TESTING.md)**.

### Code Quality

```bash
# Format code
black src/ tests/

# Lint code
ruff check src/ tests/

# Type checking
mypy src/
```

## Configuration

Configuration is managed through environment variables. See `.env.example` for all available options.

Key configuration areas:
- **API Keys** - Anthropic, Slack, GitHub, etc.
- **Databases** - PostgreSQL, Redis, Elasticsearch
- **Feature Flags** - Enable/disable specific detection types
- **Observability** - Prometheus, Grafana, Sentry

## API Documentation

Once running, visit http://localhost:8000/docs for interactive API documentation.

### Available Endpoints

#### Search API

**POST /api/v1/search/** - Semantic search across messages

Search for messages using semantic similarity with flexible filtering options.

```bash
curl -X POST http://localhost:8000/api/v1/search/ \
  -H "Content-Type: application/json" \
  -d '{
    "query": "OAuth implementation",
    "limit": 5,
    "threshold": 0.7,
    "source_types": ["slack"],
    "channels": ["#engineering"]
  }'
```

**Response:**
```json
{
  "results": [
    {
      "content": "Starting OAuth2 integration...",
      "source": "slack",
      "channel": "#engineering",
      "author": "alice@demo.com",
      "timestamp": "2024-12-01T09:00:00Z",
      "score": 0.89,
      "message_id": 123,
      "external_id": "slack_msg_abc",
      "message_metadata": {}
    }
  ],
  "total": 1,
  "query": "OAuth implementation",
  "query_time_ms": 45,
  "threshold": 0.7
}
```

**Parameters:**
- `query` (required): Search query text (1-1000 characters)
- `limit` (optional): Maximum results to return (1-100, default: 10)
- `threshold` (optional): Minimum similarity score (0.0-1.0, default: 0.0)
- `source_types` (optional): Filter by source types (e.g., ["slack", "github"])
- `channels` (optional): Filter by specific channels
- `date_from` (optional): Filter messages from this date (ISO format)
- `date_to` (optional): Filter messages until this date (ISO format)

**GET /api/v1/search/health** - Search service health check

Check the health and status of the search service and its dependencies.

```bash
curl http://localhost:8000/api/v1/search/health
```

**Response:**
```json
{
  "status": "healthy",
  "vector_store": {
    "connected": true,
    "collection": "coordination_messages",
    "document_count": 24
  }
}
```

#### Gap Detection API (Coming Soon)

**POST /api/v1/gaps/detect** - Detect coordination gaps

```bash
POST /api/v1/gaps/detect
{
  "timeframe_days": 30,
  "sources": ["slack", "github", "google_docs"],
  "gap_types": ["duplicate_work", "missing_context"],
  "min_impact_score": 0.7
}
```

*Note: Gap detection endpoints are planned for future milestones.*

## Deployment

### Docker

```bash
# Build image
docker build -t coordination-detector:latest .

# Run container
docker run -p 8000:8000 --env-file .env coordination-detector:latest
```

### Kubernetes

```bash
# Deploy to Kubernetes
kubectl apply -f k8s/

# Check status
kubectl get pods -n coordination-prod
```

### Terraform

```bash
# Infrastructure as code
cd terraform
terraform init
terraform plan
terraform apply
```

## Architecture

The system uses a multi-stage detection pipeline:

1. **Data Ingestion** - Stream events from all sources
2. **Entity Extraction** - Identify people, teams, projects
3. **Semantic Indexing** - Embed content for similarity search
4. **Pattern Detection** - Apply gap detection algorithms
5. **Impact Scoring** - Rank gaps by organizational cost
6. **LLM Reasoning** - Generate insights with Claude
7. **Alert Routing** - Notify relevant stakeholders

For detailed architecture documentation, see [CLAUDE.md](./CLAUDE.md).

## Contributing

Contributions are welcome! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Workflow

- Follow PEP 8 style guidelines
- Write tests for new features
- Update documentation as needed
- Ensure all tests pass before submitting PR

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Resources

- [Information Retrieval Fundamentals](https://nlp.stanford.edu/IR-book/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Claude API Reference](https://docs.anthropic.com/)
- [UV Package Manager](https://github.com/astral-sh/uv)

## Acknowledgments

Built with [Claude Code](https://claude.com/claude-code) for AI-native development.

---

**Status**: Alpha - Active Development
**Python Version**: 3.11+
**Last Updated**: December 2024
