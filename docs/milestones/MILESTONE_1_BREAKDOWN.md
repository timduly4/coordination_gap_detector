# Milestone 1: Foundation & Mock Data - PR Breakdown

## Overview
Breaking Milestone 1 into smaller, focused PRs makes code review easier, shows incremental progress, and creates a realistic commit history. Each PR should be independently reviewable and mergeable.

---

## Milestone 1A: Project Scaffolding & Configuration
**Branch**: `feat/project-setup`  
**Time**: ~2 hours  
**Files Changed**: ~8-10 files

### Changes:
```
coordination-gap-detector/
├── .gitignore
├── .python-version
├── pyproject.toml
├── README.md
├── CLAUDE.md
├── MILESTONES.md
├── .env.example
└── src/
    ├── __init__.py
    └── config.py
```

### Specific Tasks:
- [ ] Initialize git repository
- [ ] Create pyproject.toml with UV configuration
- [ ] Add core dependencies (fastapi, uvicorn, pydantic, python-dotenv)
- [ ] Create .gitignore (Python, venv, IDE files)
- [ ] Add .python-version (3.11)
- [ ] Create src/ directory structure
- [ ] Implement config.py with Pydantic Settings
- [ ] Create .env.example with placeholder values
- [ ] Write basic README with project description
- [ ] Add CLAUDE.md and MILESTONES.md

### PR Description Template:
```markdown
## Project Scaffolding

Initial project setup with UV, FastAPI, and configuration management.

### Changes
- Set up UV-based Python project
- Configure FastAPI application structure
- Add Pydantic settings management
- Create environment variable template

### Testing
- [ ] UV can install dependencies: `uv pip install -e .`
- [ ] No syntax errors in config.py
- [ ] .env.example contains all required variables

### Next Steps
- Add Docker Compose setup (PR 1B)
```

**Commit Messages:**
```
chore: initialize project with UV and git
feat: add pydantic settings configuration
docs: add README, CLAUDE.md, and MILESTONES.md
```

---

## Milestone 1B: Docker & Infrastructure Setup
**Branch**: `feat/docker-infrastructure`  
**Time**: ~2-3 hours  
**Files Changed**: ~5-7 files

### Changes:
```
├── docker-compose.yml
├── Dockerfile
├── .dockerignore
└── src/
    └── main.py              # Basic FastAPI app
```

### Specific Tasks:
- [ ] Create Dockerfile (multi-stage build)
- [ ] Add docker-compose.yml (Postgres, Redis, ChromaDB services)
- [ ] Create .dockerignore
- [ ] Implement basic FastAPI app in main.py
- [ ] Add health check endpoint
- [ ] Configure service networking in docker compose
- [ ] Add volume mounts for development

### PR Description Template:
```markdown
## Docker Infrastructure

Docker Compose setup with Postgres, Redis, and ChromaDB for local development.

### Changes
- Multi-stage Dockerfile for production builds
- Docker Compose with all required services
- Basic FastAPI application with health check
- Development volume mounts for hot reload

### Testing
- [ ] `docker compose up` starts all services
- [ ] `curl http://localhost:8000/health` returns 200
- [ ] Services can communicate (pg, redis, chroma accessible)
- [ ] Hot reload works when editing src/ files

### Services
- API: http://localhost:8000
- Postgres: localhost:5432
- Redis: localhost:6379
- ChromaDB: localhost:8001
```

**Commit Messages:**
```
chore: add Dockerfile with multi-stage build
feat: add docker compose with postgres, redis, chromadb
feat: create basic FastAPI app with health endpoint
```

---

## Milestone 1C: Database Models & Connection
**Branch**: `feat/database-setup`  
**Time**: ~2-3 hours  
**Files Changed**: ~6-8 files

### Changes:
```
src/
├── db/
│   ├── __init__.py
│   ├── postgres.py          # SQLAlchemy setup
│   ├── models.py            # DB models
│   └── vector_store.py      # ChromaDB client (stub)
└── api/
    └── dependencies.py      # FastAPI dependencies
```

### Specific Tasks:
- [ ] Add SQLAlchemy to dependencies
- [ ] Create database connection management
- [ ] Define initial data models (Message, Source)
- [ ] Add Alembic for migrations
- [ ] Create first migration
- [ ] Implement ChromaDB client (basic connection)
- [ ] Add database dependency injection for FastAPI
- [ ] Add database health check

### PR Description Template:
```markdown
## Database Setup

PostgreSQL with SQLAlchemy and ChromaDB client configuration.

### Changes
- SQLAlchemy models for messages and sources
- Database connection management with dependency injection
- Alembic migrations setup
- ChromaDB client initialization
- Enhanced health check with DB status

### Testing
- [ ] Alembic migrations run successfully
- [ ] Database tables created correctly
- [ ] Health check shows DB connectivity
- [ ] ChromaDB client connects

### Database Schema
- `messages` table: id, source_id, content, timestamp, metadata
- `sources` table: id, type, name, config
```

**Commit Messages:**
```
feat: add sqlalchemy models and postgres connection
feat: configure alembic for database migrations
feat: add chromadb client initialization
feat: implement database dependency injection
```

---

## Milestone 1D: Mock Data Generator
**Branch**: `feat/mock-data-generator`  
**Time**: ~3-4 hours  
**Files Changed**: ~4-6 files

### Changes:
```
scripts/
├── generate_mock_data.py
└── __init__.py
src/
└── ingestion/
    └── slack/
        ├── __init__.py
        └── mock_client.py
tests/
└── test_mock_data.py
```

### Specific Tasks:
- [ ] Create mock Slack message data structures
- [ ] Implement MockSlackClient with realistic data
- [ ] Generate 3-4 conversation scenarios
- [ ] Create script to seed database with mock data
- [ ] Add variety (threads, reactions, different channels)
- [ ] Include temporal patterns (conversations over time)
- [ ] Write tests for mock data generation
- [ ] Document mock data scenarios in README

### PR Description Template:
```markdown
## Mock Data Generator

Realistic Slack message data for development and testing.

### Changes
- MockSlackClient with conversation scenarios
- Script to generate and seed mock data
- Multiple conversation types (technical discussions, decisions, threads)
- Tests for mock data generation

### Scenarios Included
1. **OAuth Implementation Discussion** - Technical thread with 8 messages
2. **Team Decision Thread** - Decision-making with stakeholders
3. **Bug Report Discussion** - Issue identification and resolution
4. **Feature Planning** - Cross-team feature discussion

### Testing
- [ ] `python scripts/generate_mock_data.py` seeds database
- [ ] Mock data includes realistic timestamps
- [ ] Messages have proper threading structure
- [ ] Different channels and participants represented

### Usage
\`\`\`bash
# Generate and load mock data
uv run python scripts/generate_mock_data.py --scenarios all

# Verify in database
docker compose exec postgres psql -U user -d coordination -c "SELECT COUNT(*) FROM messages;"
\`\`\`
```

**Commit Messages:**
```
feat: implement MockSlackClient with conversation scenarios
feat: add mock data generation script
test: add tests for mock data scenarios
docs: document mock data scenarios
```

---

## Milestone 1E: Vector Store & Embedding
**Branch**: `feat/vector-embeddings`  
**Time**: ~3-4 hours  
**Files Changed**: ~5-7 files

### Changes:
```
src/
├── db/
│   └── vector_store.py      # Complete ChromaDB implementation
├── models/
│   ├── __init__.py
│   ├── embeddings.py        # Embedding generation
│   └── schemas.py           # Pydantic models
└── utils/
    └── text_processing.py   # Text utilities
```

### Specific Tasks:
- [ ] Complete ChromaDB vector store implementation
- [ ] Add embedding generation (Claude API or sentence-transformers)
- [ ] Implement document chunking for messages
- [ ] Create Pydantic schemas for API requests/responses
- [ ] Add vector store operations (insert, search)
- [ ] Implement similarity search
- [ ] Add error handling and logging
- [ ] Write tests for vector operations

### PR Description Template:
```markdown
## Vector Store & Embeddings

ChromaDB integration with semantic search capability.

### Changes
- Complete ChromaDB vector store implementation
- Embedding generation using sentence-transformers
- Document chunking and indexing
- Similarity search with configurable parameters
- Pydantic schemas for type safety

### Features
- Semantic search over messages
- Configurable similarity threshold
- Metadata filtering support
- Batch embedding operations

### Testing
- [ ] Can insert messages into vector store
- [ ] Similarity search returns relevant results
- [ ] Embeddings are deterministic
- [ ] Handles edge cases (empty queries, no results)

### Usage
\`\`\`python
from src.db.vector_store import VectorStore

store = VectorStore()
results = store.search("OAuth implementation", limit=5)
# Returns top 5 most similar messages
\`\`\`
```

**Commit Messages:**
```
feat: implement chromadb vector store operations
feat: add embedding generation with sentence-transformers
feat: create pydantic schemas for API models
test: add vector store integration tests
```

---

## Milestone 1F: Search API Endpoint
**Branch**: `feat/search-endpoint`  
**Time**: ~3-4 hours  
**Files Changed**: ~6-8 files

### Changes:
```
src/
├── api/
│   ├── __init__.py
│   └── routes/
│       ├── __init__.py
│       └── search.py        # Search endpoint
├── services/
│   ├── __init__.py
│   └── search_service.py    # Business logic
└── main.py                  # Updated with router
tests/
├── conftest.py              # Pytest fixtures
└── test_api/
    └── test_search.py
```

### Specific Tasks:
- [ ] Create search API endpoint
- [ ] Implement SearchService with business logic
- [ ] Add request/response validation
- [ ] Implement error handling
- [ ] Add logging for requests
- [ ] Write API tests
- [ ] Add endpoint documentation (OpenAPI)
- [ ] Update health check with service status

### PR Description Template:
```markdown
## Search API Endpoint

RESTful search endpoint with semantic similarity.

### Changes
- POST /api/v1/search endpoint
- SearchService for business logic separation
- Comprehensive error handling
- Request/response validation with Pydantic
- API tests with pytest
- OpenAPI documentation

### API Example
\`\`\`bash
curl -X POST http://localhost:8000/api/v1/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "OAuth implementation",
    "limit": 5,
    "threshold": 0.7
  }'
\`\`\`

### Response
\`\`\`json
{
  "results": [
    {
      "content": "Starting work on OAuth2 integration...",
      "source": "slack",
      "channel": "#platform",
      "author": "alice@demo.com",
      "timestamp": "2024-12-01T09:00:00Z",
      "score": 0.89
    }
  ],
  "total": 5,
  "query_time_ms": 45
}
\`\`\`

### Testing
- [ ] Endpoint returns 200 for valid queries
- [ ] Handles invalid requests (400)
- [ ] Returns properly formatted results
- [ ] Respects limit parameter
- [ ] Filters by similarity threshold
```

**Commit Messages:**
```
feat: add search API endpoint
feat: implement SearchService with business logic
test: add comprehensive API tests for search
docs: add OpenAPI documentation for search endpoint
```

---

## Milestone 1G: Testing & Documentation
**Branch**: `feat/testing-and-docs`  
**Time**: ~2-3 hours  
**Files Changed**: ~8-10 files

### Changes:
```
tests/
├── __init__.py
├── conftest.py              # Enhanced fixtures
├── test_search.py           # Unit tests
├── test_integration/
│   └── test_search_flow.py  # Integration tests
└── test_db/
    ├── test_vector_store.py
    └── test_postgres.py
README.md                     # Updated with examples
docs/
└── API.md                   # API documentation
```

### Specific Tasks:
- [ ] Add comprehensive unit tests (80%+ coverage)
- [ ] Create integration tests
- [ ] Add pytest fixtures for common test data
- [ ] Configure pytest settings (pytest.ini)
- [ ] Update README with complete setup instructions
- [ ] Create API documentation
- [ ] Add examples and usage guide
- [ ] Document mock data scenarios
- [ ] Add troubleshooting section

### PR Description Template:
```markdown
## Testing & Documentation

Comprehensive tests and documentation for Milestone 1.

### Changes
- Unit tests for all major components
- Integration tests for end-to-end flows
- Enhanced pytest fixtures
- Updated README with setup guide
- API documentation
- Usage examples

### Test Coverage
- Vector store operations: 95%
- Search service: 90%
- API endpoints: 88%
- Overall: 85%

### Testing
- [ ] All tests pass: `pytest`
- [ ] Coverage meets threshold: `pytest --cov=src --cov-report=html`
- [ ] Integration tests work: `pytest tests/test_integration/`

### Documentation Updates
- Step-by-step setup instructions
- API endpoint documentation
- Mock data scenarios explained
- Troubleshooting guide
- Development workflow
```

**Commit Messages:**
```
test: add comprehensive unit tests for all components
test: add integration tests for search flow
docs: update README with setup and usage guide
docs: add API documentation
```

---

## Summary: Milestone 1 PRs

| PR | Branch | Time | Files | Focus |
|----|--------|------|-------|-------|
| **1A** | `feat/project-setup` | 2h | ~10 | Project structure, config |
| **1B** | `feat/docker-infrastructure` | 2-3h | ~7 | Docker, services, FastAPI |
| **1C** | `feat/database-setup` | 2-3h | ~8 | SQLAlchemy, migrations |
| **1D** | `feat/mock-data-generator` | 3-4h | ~6 | Mock Slack data |
| **1E** | `feat/vector-embeddings` | 3-4h | ~7 | ChromaDB, embeddings |
| **1F** | `feat/search-endpoint` | 3-4h | ~8 | Search API |
| **1G** | `feat/testing-and-docs` | 2-3h | ~10 | Tests, documentation |

**Total**: 17-25 hours across 7 PRs

## PR Workflow

### For Each PR:

1. **Create branch from main**
```bash
git checkout main
git pull origin main
git checkout -b feat/project-setup
```

2. **Make changes & commit frequently**
```bash
git add .
git commit -m "feat: add pydantic settings configuration"
# ... more commits ...
```

3. **Push and create PR**
```bash
git push origin feat/project-setup
# Create PR on GitHub
```

4. **Self-review checklist**
- [ ] Code runs without errors
- [ ] Tests pass
- [ ] No unnecessary files committed
- [ ] Commit messages are clear
- [ ] PR description is complete

5. **Merge and tag (if completing milestone)**
```bash
git checkout main
git merge feat/project-setup
git tag v0.1.0-alpha.1  # Or v0.1.0 after PR 1G
git push origin main --tags
```

## Benefits of This Approach

✅ **Easier to review** - Each PR has a clear, focused purpose  
✅ **Shows progression** - Realistic development history  
✅ **Safe to experiment** - Can abandon a PR without losing other work  
✅ **Better for resume** - Shows professional Git workflow  
✅ **Easier to debug** - Can bisect issues to specific PRs  
✅ **Demonstrates skills** - Shows you can break down work effectively

## Optional: Add PR Templates

Create `.github/pull_request_template.md`:

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] New feature
- [ ] Bug fix
- [ ] Documentation
- [ ] Refactoring

## Testing
- [ ] All tests pass
- [ ] Added new tests
- [ ] Manual testing completed

## Checklist
- [ ] Code follows project style
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No console warnings/errors
```

## Git Commit Convention

Use conventional commits:
- `feat:` - New feature
- `fix:` - Bug fix
- `docs:` - Documentation only
- `test:` - Adding tests
- `refactor:` - Code restructuring
- `chore:` - Maintenance tasks

Examples:
```
feat: add chromadb vector store implementation
fix: handle empty search queries gracefully
docs: update README with docker setup instructions
test: add integration tests for search flow
refactor: extract embedding logic into separate module
chore: update dependencies to latest versions
```